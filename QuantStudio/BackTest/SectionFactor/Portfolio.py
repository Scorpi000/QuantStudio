# coding=utf-8
import datetime as dt
import base64
from io import BytesIO
from typing import Optional, Dict, Literal, List, Any

import numpy as np
import pandas as pd
from pydantic import Field
import statsmodels.api as sm
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdate

from QuantStudio.Core import __QS_Error__
from QuantStudio.Core.QSObject import Panel
import QuantStudio.Factor.FactorOperator as fo
from QuantStudio.Factor.FactorOperation import PanelOperator, SectionOperator
from QuantStudio.Factor.Factor import Factor, FactorInitData, FactorContext
from QuantStudio.BackTest.BackTestModel import BTNode, BTLocalContext, BTInitData
from QuantStudio.BackTest.SectionFactor.IC import _QS_formatMatplotlibPercentage, _QS_formatPandasPercentage
from QuantStudio.Tools.StrategyTestFun import calcMaxDrawdownRate, calcLSYield, testPortfolioStrategy_pd


def _QS_plotStatistics(axes, x_data, x_ticklabels, left_data, left_formatter, right_data=None, right_formatter=None, right_axes=True, title=None):
    axes.yaxis.set_major_formatter(left_formatter)
    axes.bar(x_data, left_data.values, label=left_data.name, color="steelblue")
    if right_data is not None:
        if right_axes:
            axes.legend(loc='upper left')
            right_axes = axes.twinx()
            right_axes.yaxis.set_major_formatter(right_formatter)
            right_axes.plot(x_data, right_data.values, label=right_data.name, color="indianred", lw=2.5)
            right_axes.legend(loc="upper right")
        else:
            axes.plot(x_data, right_data.values, label=right_data.name, color="indianred", lw=2.5)
            axes.legend(loc='best')
    else:
        axes.legend(loc='best')
    axes.set_xticks(x_data)
    axes.set_xticklabels(x_ticklabels)
    if title is not None:
        axes.set_title(title)
    return axes


# 筛选投资组合
class CalcMaskPortfolio(SectionOperator):
    """筛选投资组合"""
    def __init__(self, descriptor_ids=None, args={}, config_file=None, **kwargs):
        Arity = args.get("Arity", None) or 1
        Args = {"Name": "calcMaskPortfolio"} | args | {"DTMode": "多时点", "OutputMode": "全截面", "DataType": "double"}
        Args["DescriptorSection"] = [Args.get("DescriptorSection", [descriptor_ids])[0]] * Arity
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f, idt, iid, x, args):
        SectionIDs = (self._QSArgs.DescriptorSection[0] if self._QSArgs.DescriptorSection[0] else iid)
        Mask, x = pd.DataFrame(x[0]==1, index=idt, columns=SectionIDs), x[1:]
        if f._QSArgs.ModelArgs["weight"]:
            Weight, x = pd.DataFrame(x[0], index=idt, columns=SectionIDs), x[1:]
        else:
            Weight = pd.DataFrame(1, index=idt, columns=SectionIDs)
        if f._QSArgs.ModelArgs["cat_data"]:
            CatData, x = pd.DataFrame(x[0], index=idt, columns=SectionIDs), x[1:]
            CatData = CatData.where(CatData.notnull(), "None")
            if f._QSArgs.ModelArgs["cat_weight"]:
                CatWeight = pd.DataFrame(x[-1], index=idt, columns=SectionIDs)
            else:
                CatWeight = pd.DataFrame(1, index=idt, columns=SectionIDs)
        if f._QSArgs.CalcDTRuler:
            RebalanceDTs = sorted(set(idt).intersection(f._QSArgs.CalcDTRuler))
            Mask = Mask.reindex(index=RebalanceDTs).fillna(False)
            Weight = Weight.reindex(index=RebalanceDTs)
            if f._QSArgs.ModelArgs["cat_data"]:
                CatData = CatData.reindex(index=RebalanceDTs)
                CatWeight = CatWeight.reindex(index=RebalanceDTs)
        if not f._QSArgs.ModelArgs["cat_data"]:
            Porftolio = Weight.where(Mask, np.nan)
            Porftolio = (Porftolio.T / Porftolio.sum(axis=1)).T
            return Porftolio.reindex(index=idt, columns=iid).values
        else:
            #Rslt = pd.DataFrame({"mask": Mask.stack(), "weight": Weight.stack(), "cat_data": CatData.stack(), "cat_weight": CatWeight.stack()}).reset_index()
            Rslt = Panel({"mask": Mask, "weight": Weight, "cat_data": CatData, "cat_weight": CatWeight}).to_frame().reset_index()
            Rslt.columns = ["dt", "id"] + Rslt.columns[2:].tolist()
            if not Rslt["mask"].any(): return np.full(shape=(len(idt), len(iid)), fill_value=np.nan, dtype=float)
            Tmp = Rslt.groupby(["dt", "cat_data"])[["cat_weight"]].sum().reset_index()
            if not f._QSArgs.ModelArgs["cat_weight"]: Tmp["cat_weight"] = 1
            Tmp = pd.merge(Tmp, Tmp.groupby(["dt"])["cat_weight"].sum().to_frame("total_cat_weight"), how="left", left_on=["dt"], right_index=True)
            Tmp["cat_weight"] = Tmp["cat_weight"] / Tmp["total_cat_weight"]
            Rslt = Rslt[Rslt["mask"]]
            Rslt = pd.merge(Rslt, Tmp.loc[:, ["dt", "cat_data", "cat_weight"]], how="left", left_on=["dt", "cat_data"], right_on=["dt", "cat_data"], suffixes=("", "_total"))
            Rslt = pd.merge(Rslt, Rslt.groupby(["dt", "cat_data"])[["weight"]].sum(), how="left", left_on=["dt", "cat_data"], right_index=True, suffixes=("", "_total"))
            Rslt["weight"] = Rslt["weight"] / Rslt["weight_total"] * Rslt["cat_weight_total"]
            return Rslt.set_index(["dt", "id"])["weight"].unstack().reindex(index=idt, columns=iid).values
    
    def __call__(self, mask:Factor, weight:Optional[Factor]=None, cat_data:Optional[Factor]=None, cat_weight:Optional[Factor]=None, factor_args:Dict={}, **kwargs):
        Factors = [mask]
        if weight is not None: Factors.append(weight)
        if cat_data is not None:
            Factors.append(cat_data)
            if cat_weight is not None:
                Factors.append(cat_weight)
        factor_args["ModelArgs"] = factor_args.get("ModelArgs", {}) | {"weight": (weight is not None), "cat_data": (cat_data is not None), "cat_weight": (cat_weight is not None)}
        return super().__call__(*Factors, factor_args=factor_args, **kwargs)


# 分位数组合
def makeQuantilePortfolio(factor:Factor, mask:Optional[Factor]=None, cat_data:Optional[Factor]=None, weight:Optional[Factor]=None, descriptor_ids:Optional[List[str]]=None, rebalance_dts:Optional[List[dt.datetime]]=None, ascending:bool=False, group_num:int=5, **kwargs) -> List[Factor]:
    rank = fo.SectionRank(ascending=ascending, uniformization=True)
    Rank = rank(factor, mask=mask, cat_data=cat_data, factor_args={"CalcDTRuler": rebalance_dts})
    if weight is None: weight = 1
    calcMaskPortfolio = CalcMaskPortfolio(descriptor_ids=descriptor_ids, **kwargs)
    Portfolio = []
    for i in range(group_num):
        iLeft, iRight = i / group_num, (i+1) / group_num
        iPortfolio = calcMaskPortfolio((Rank >= iLeft) & (Rank < iRight), weight=weight)
        Portfolio.append(iPortfolio)
    return Portfolio


# 计算投资组合净值
class CalcPortfolioNV(PanelOperator):
    def __init__(self, if_price_missing:Literal["沿用前值", "填充为0"]="沿用前值", descriptor_ids=None, args={}, config_file=None, **kwargs):
        Args = {"Name": "calcPortfolioNV"} | args | {"Arity": 4, "DTMode": "多时点", "OutputMode": "全截面", "DataType": "double", "LookBack": [1, 0, 0, 0], "iInitFactor": 0, "LookBackMode": ["扩张窗口"] * 4, "StartDT": [None] * 4}
        Args["ModelArgs"] = {"if_price_missing": if_price_missing} | Args.get("ModelArgs", {})
        Args["DescriptorSection"] = [None] + [descriptor_ids] * 3
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f, idt, iid, x, args):
        SectionIDs = (self._QSArgs.DescriptorSection[1] if self._QSArgs.DescriptorSection[1] else iid)
        Portfolio = pd.DataFrame(x[1], index=idt[1:], columns=SectionIDs).dropna(how="all")
        Price = pd.DataFrame(x[2], index=idt[1:], columns=SectionIDs)
        FeeRate = pd.DataFrame(x[3], index=idt[1:], columns=SectionIDs).reindex(index=Portfolio.index)# TODO
        if args["if_price_missing"]=="沿用前值": Price = Price.ffill()
        NV = testPortfolioStrategy_pd(Portfolio.dropna(how="all"), Price)
        if pd.isnull(NV.iloc[0]):
            NV.iloc[0] = 1
            NV = NV.ffill()
        return np.reshape(NV.values, (-1, 1)).repeat(len(iid), axis=1) * x[0][0]

    def __call__(self, portfolio:Factor, price:Factor, init_nv=1, fee_rate=0, factor_args:Dict={}, **kwargs):
        return super().__call__(init_nv, portfolio, price, fee_rate, factor_args=factor_args, **kwargs)


class MultiPortfolio(BTNode):
    """多组合对比"""
    class __QS_ArgClass__(BTNode.__QS_ArgClass__):
        Name: str = Field(default="多组合对比", frozen=True, title="名称")
        RebalanceDTs: Optional[List[dt.datetime]] = Field(default=None, title="再平衡时点", frozen=True)

    def __init__(self, nv_list:list, bmk_nv=None, portfolio_list:Optional[list]=None, bmk_portfolio=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        self._NVList = nv_list
        self._BmkNV = bmk_nv
        self._PortfolioList = portfolio_list
        self._BmkPortfolio = bmk_portfolio
        Deps = nv_list.copy()
        if bmk_nv is not None: Deps.append(bmk_nv)
        if portfolio_list:
            if len(nv_list)!=len(portfolio_list):
                raise __QS_Error__(f"净值列表的长度({len(nv_list)})不等于投资组合列表的长度({len(portfolio_list)})")
            Deps += portfolio_list
        if (bmk_nv is not None) and (bmk_portfolio is not None): Deps.append(bmk_portfolio)
        super().__init__(deps=Deps, args=args, config_file=config_file, **kwargs)
        PortfolioNameList = [iF.Name for iF in self._NVList]
        nPortfolio = len(PortfolioNameList)
        if len(set(PortfolioNameList)) < nPortfolio:
            self._QS_Logger.warning(f"输入的投资组合净值列表的名称有重复: {PortfolioNameList}, 将附加序号前缀")
            nPos = 1 if nPortfolio<=1 else int(np.log10(nPortfolio - 1)) + 1
            PortfolioNameList = [f"{str(i).zfill(nPos)}-{iName}" for i, iName in enumerate(PortfolioNameList)]
        if self._BmkPortfolio is not None: PortfolioNameList.append("基准")
        self._PortfolioNameList = PortfolioNameList

    def genMatplotlibFig(self, output, file_path=None):
        nRow, nCol = 3, 3
        Fig = Figure(figsize=(min(32, 16+(nCol-1)*8), 8*nRow))
        GroupNum = self._Output["超额净值"].shape[1]
        xData = np.arange(1, GroupNum + 1)
        xTickLabels = [str(iInd) for iInd in self._Output["统计数据"].index[:GroupNum]]
        PercentageFormatter = FuncFormatter(_QS_formatMatplotlibPercentage)
        FloatFormatter = FuncFormatter(lambda x, pos: '%.2f' % (x, ))
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 1), xData, xTickLabels, self._Output["统计数据"]["年化超额收益率"].iloc[:GroupNum], PercentageFormatter, self._Output["统计数据"]["胜率"].iloc[:GroupNum], PercentageFormatter)
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 2), xData, xTickLabels, self._Output["统计数据"]["信息比率"].iloc[:GroupNum], PercentageFormatter, None)
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 3), xData, xTickLabels, self._Output["统计数据"]["超额最大回撤率"].iloc[:GroupNum], PercentageFormatter, None)
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 4), xData, xTickLabels, self._Output["统计数据"]["年化收益率"].iloc[:GroupNum], PercentageFormatter, pd.Series(self._Output["统计数据"].loc["Bmk", "年化收益率"], index=self._Output["统计数据"].index[:GroupNum], name="基准"), PercentageFormatter, False)
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 5), xData, xTickLabels, self._Output["统计数据"]["Sharpe比率"].iloc[:GroupNum], FloatFormatter, pd.Series(self._Output["统计数据"].loc["Bmk", "Sharpe比率"], index=self._Output["统计数据"].index[:GroupNum], name="基准"), FloatFormatter, False)
        _QS_plotStatistics(Fig.add_subplot(nRow, nCol, 6), xData, xTickLabels, self._Output["统计数据"]["平均换手率"].iloc[:GroupNum], PercentageFormatter, None)
        Axes = Fig.add_subplot(nRow, nCol, 7)
        Axes.xaxis_date()
        Axes.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
        for i in range(GroupNum):
            Axes.plot(self._Output["超额净值"].index, self._Output["超额净值"].iloc[:, i].values, label=str(self._Output["超额净值"].columns[i]), lw=2.5)
        Axes.legend(loc='best')
        Axes.set_title("超额净值")
        Axes = Fig.add_subplot(nRow, nCol, 8)
        xData = np.arange(0, self._Output["净值"].shape[0])
        xTicks = np.arange(0, self._Output["净值"].shape[0], max(1, int(self._Output["净值"].shape[0]/8)))
        xTickLabels = [self._Output["净值"].index[i].strftime("%Y-%m-%d") for i in xTicks]
        Axes.plot(xData, self._Output["净值"]["Top-Bottom"].values, label="Top-Bottom 净值", color="indianred", lw=2.5)
        Axes.legend(loc='upper left')
        Axes.set_title("Top-Bottom 多空组合")
        RAxes = Axes.twinx()
        RAxes.yaxis.set_major_formatter(PercentageFormatter)
        RAxes.bar(xData, self._Output["收益率"]["Top-Bottom"].values, label="Top-Bottom 收益率", color="steelblue")
        RAxes.legend(loc="upper right")
        Axes.set_xticks(xTicks)
        Axes.set_xticklabels(xTickLabels)
        Axes = Fig.add_subplot(nRow, nCol, 9)
        Axes.xaxis_date()
        Axes.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
        for i in range(GroupNum+1):
            Axes.plot(self._Output["净值"].index, self._Output["净值"].iloc[:, i].values, label=str(self._Output["净值"].columns[i]), lw=2.5)
        Axes.legend(loc='best')
        Axes.set_title("多头净值")
        if file_path is not None: Fig.savefig(file_path, dpi=150, bbox_inches='tight')
        return Fig
    
    def genReport(self, output):
        HTML = "参数设置: "
        HTML += '<ul align="left">'
        if isinstance(getattr(self.Deps[0], "Operator", None), CalcPortfolioNV):
            ModelArgs = self.Deps[0].Operator._QSArgs.ModelArgs
            HTML += f"<li>相关性方法: {ModelArgs['corr_method']}</li>"
            HTML += f"<li>回溯期数: {ModelArgs['period_lookback']}</li>"
        if self._QSArgs.RebalanceDTs is not None:
            HTML += "<li>再平衡时点: 自定义时点</li>"
        else:
            HTML += "<li>再平衡时点: 所有时点</li>"
        HTML += f"<li>移动平均期数: {self._QSArgs.RollingAvgPeriod}</li>"
        HTML += "</ul>"
        Formatters = [_QS_formatPandasPercentage]*3+[lambda x:'{0:.2f}'.format(x)]*2+[_QS_formatPandasPercentage]*2+[lambda x: x.strftime("%Y-%m-%d") if pd.notnull(x) else "NaT"]*2
        Formatters += [_QS_formatPandasPercentage]*3+[lambda x:'{0:.2f}'.format(x)]*2+[_QS_formatPandasPercentage]*2+[lambda x: x.strftime("%Y-%m-%d") if pd.notnull(x) else "NaT"]*2
        Formatters += [lambda x:'{0:.2f}'.format(x)]*2
        iHTML = output["统计数据"].to_html(formatters=Formatters)
        Pos = iHTML.find(">")
        HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
        Fig = self.genMatplotlibFig(output)
        # figure 保存为二进制文件
        Buffer = BytesIO()
        Fig.savefig(Buffer, bbox_inches='tight')
        PlotData = Buffer.getvalue()
        # 图像数据转化为 HTML 格式
        ImgStr = "data:image/png;base64,"+base64.b64encode(PlotData).decode()
        HTML += ('<img src="%s">' % ImgStr)
        return HTML

    def _QS_calcStats(self, output):
        nDT = output["净值"].shape[0] - 1
        nDays = (output["净值"].index[-1] - output["净值"].index[0]).days
        nYear = nDays / 365
        TotalReturn = output["净值"].iloc[-1, :] - 1
        output["统计数据"] = pd.DataFrame(index=TotalReturn.index)
        output["统计数据"]["总收益率"] = TotalReturn
        output["统计数据"]["年化收益率"] = (1 + TotalReturn) ** (1 / nYear) - 1
        output["统计数据"]["波动率"] = output["收益率"].std() * np.sqrt(nDT / nYear)
        output["统计数据"]["Sharpe比率"] = output["统计数据"]["年化收益率"] / output["统计数据"]["波动率"]
        output["统计数据"]["t统计量(Sharpe比率)"] = (output["统计数据"]["Sharpe比率"] - output["统计数据"]["Sharpe比率"]["基准"]) / np.sqrt(2/nYear)
        output["统计数据"]["平均换手率"] = output["换手率"].mean()
        output["统计数据"]["最大回撤率"] = pd.Series(np.nan,index=output["统计数据"].index)
        output["统计数据"]["最大回撤开始时间"] = pd.Series(index=output["统计数据"].index,dtype="O")
        output["统计数据"]["最大回撤结束时间"] = pd.Series(index=output["统计数据"].index,dtype="O")
        for iCol in output["净值"].columns:
            iMaxDD,iStartPos,iEndPos = calcMaxDrawdownRate(output["净值"].loc[:, iCol].values)
            output["统计数据"].loc[iCol, "最大回撤率"] = abs(iMaxDD)
            output["统计数据"].loc[iCol, "最大回撤开始时间"] = (output["净值"].index[iStartPos] if iStartPos is not None else None)
            output["统计数据"].loc[iCol, "最大回撤结束时间"] = (output["净值"].index[iEndPos] if iEndPos is not None else None)
        output["统计数据"]["超额收益率"] = output["超额净值"].iloc[-1, :] - 1
        output["统计数据"]["年化超额收益率"] = (1 + output["统计数据"]["超额收益率"]) ** (1/nYear) - 1
        output["统计数据"]["跟踪误差"] = output["超额收益率"].std() * np.sqrt(nDT/nYear)
        output["统计数据"]["信息比率"] = output["统计数据"]["年化超额收益率"] / output["统计数据"]["跟踪误差"]
        output["统计数据"]["t统计量(信息比率)"] = output["统计数据"]["信息比率"] * np.sqrt(nYear)
        output["统计数据"]["胜率"] = (output["超额收益率"] > 0).sum() / nDT
        output["统计数据"]["超额最大回撤率"] = pd.Series(np.nan, index=output["统计数据"].index)
        output["统计数据"]["超额最大回撤开始时间"] = pd.Series(index=output["统计数据"].index, dtype="O")
        output["统计数据"]["超额最大回撤结束时间"] = pd.Series(index=output["统计数据"].index, dtype="O")
        for iCol in output["超额净值"].columns:
            iMaxDD, iStartPos, iEndPos = calcMaxDrawdownRate(output["超额净值"].loc[:, iCol].values)
            output["统计数据"].loc[iCol, "超额最大回撤率"] = abs(iMaxDD)
            output["统计数据"].loc[iCol, "超额最大回撤开始时间"] = (output["超额净值"].index[iStartPos] if iStartPos is not None else None)
            output["统计数据"].loc[iCol, "超额最大回撤结束时间"] = (output["超额净值"].index[iEndPos] if iEndPos is not None else None)
        output["统计数据"]["CAPM Alpha"], output["统计数据"]["CAPM Beta"] = 0.0, 0.0
        xData = sm.add_constant(output["收益率"]["基准"].values, prepend=True)
        for iCol in output["收益率"].columns:
            yData = output["收益率"][iCol].values
            try:
                Result = sm.OLS(yData, xData, missing="drop").fit()
                output["统计数据"].loc[iCol, "CAPM Beta"] = Result.params[1]
                output["统计数据"].loc[iCol, "CAPM Alpha"] = Result.params[0]
            except:
                output["统计数据"].loc[iCol, "CAPM Beta"] = np.nan
                output["统计数据"].loc[iCol, "CAPM Alpha"] = np.nan
        return output

    def init_compute(self, path: List[str], init_data: BTInitData, context: FactorContext) -> List[FactorInitData]:
        InitData = super().init_compute(path=path, init_data=init_data, context=context)
        return [FactorInitData(DTRange=iInitData.DTRange, SectionIDs=self.Deps[i].getID()) for i, iInitData in enumerate(InitData)]
    
    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[BTLocalContext]=None) -> dict:
        nPortfolio = len(self._NVList)
        PortfolioNV = pd.DataFrame({i: BwdData.iloc[:, 0] for i, BwdData in enumerate(bwd_data_list[:nPortfolio])}).sort_index(axis=1)
        PortfolioNV.columns = self._PortfolioNameList[:nPortfolio]
        BmkNV = (bwd_data_list[nPortfolio].iloc[:, 0] if self._BmkNV is not None else pd.Series(1, index=PortfolioNV.index))
        Portfolio = {self._PortfolioNameList[i]: iBwdData for i, iBwdData in enumerate(bwd_data_list[nPortfolio + int(self._BmkNV is not None):])}
        if self._QSArgs.RebalanceDTs is None:
            RebalanceIdx = None
        else:
            RebalanceDTs = sorted(PortfolioNV.index.intersection(self._QSArgs.RebalanceDTs))
            Portfolio = {iName: iPortfolio.reindex(index=RebalanceDTs) for iName, iPortfolio in Portfolio.items()}
            RebalanceIdx = pd.Series(np.arange(PortfolioNV.shape[0]), index=PortfolioNV.index)
            RebalanceIdx = sorted(RebalanceIdx[RebalanceDTs])
        Output = {"投资组合": Portfolio}
        Output["换手率"] = pd.DataFrame({iName: iPortfolio.diff().abs().sum(axis=1) for iName, iPortfolio in Portfolio.items()}).reindex(columns=self._PortfolioNameList)
        Output["净值"] = PortfolioNV
        Output["净值"]["基准"] = BmkNV
        Output["收益率"] = Output["净值"].pct_change()
        Output["超额收益率"] = Output["收益率"].iloc[:, :-1].copy()
        Output["超额净值"] = Output["超额收益率"].copy()
        for iCol in Output["超额收益率"].columns:
            Output["超额收益率"][iCol] = calcLSYield(Output["超额收益率"][iCol].values, Output["收益率"]["基准"].values, rebalance_index=RebalanceIdx)
            Output["超额净值"][iCol] = (1 + Output["超额收益率"][iCol]).cumprod()
        Output["收益率"]["Top-Bottom"] = calcLSYield(Output["收益率"].iloc[:, 0].values, Output["收益率"].iloc[:, -1].values, rebalance_index=RebalanceIdx)
        Output["净值"]["Top-Bottom"] = (1 + Output["收益率"]["Top-Bottom"]).cumprod()
        Output = self._QS_calcStats(Output)
        return Output
