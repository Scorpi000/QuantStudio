# coding=utf-8
import datetime as dt
import base64
from io import BytesIO
from typing import Optional, Literal, List, Any

import numpy as np
import pandas as pd
from numpy.lib.recfunctions import unstructured_to_structured
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
import statsmodels.api as sm
from pydantic import Field

from QuantStudio.Core import __QS_Error__
from QuantStudio.Core.QSObject import Panel
from QuantStudio.Factor.Factor import Factor, FactorContext, FactorInitData
from QuantStudio.Factor.FactorOperation import PanelOperator, PanelOperation
from QuantStudio.BackTest.BackTestModel import BTLocalContext, BTNode, BTInitData
from QuantStudio.Tools.DataPreprocessingFun import prepareRegressData


def _QS_formatMatplotlibPercentage(x, pos):
    return '%.2f%%' % (x*100, )

def _QS_formatPandasPercentage(x):
    return '{0:.2f}%'.format(x*100)

class CalcIC(PanelOperator):
    """IC 算子
    IC: 往期因子值和当期收益率的截面秩相关性。通常采用的相关系数为 Spearman 相关系数。
    """

    def __init__(self, lookback:int = 31, period_lookback:int=1, corr_method:Literal["spearman", "pearson", "kendall"]="spearman", descriptor_ids:Optional[List[str]]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Arity = args.get("Arity", None) or 1
        Args = {"Name": "calcIC"} | args | {"DTMode": "多时点", "OutputMode": "全截面", "DataType": "object"}
        Args["ModelArgs"] = {"corr_method": corr_method, "period_lookback": period_lookback} | Args.get("ModelArgs", {})
        Args["DescriptorSection"] = [Args.get("DescriptorSection", [descriptor_ids])[0]] * Arity
        Args["LookBack"] = [Args.get("LookBack", [lookback])[0]] * Arity
        Args["CompoundType"] = [("IC", float), ("Breadth", float)]
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        SectionIDs = (self._QSArgs.DescriptorSection[0] if self._QSArgs.DescriptorSection[0] else iid)
        Price, x = pd.DataFrame(x[0].T, columns=idt, index=SectionIDs), x[1:]
        if f._QSArgs.CalcDTRuler:
            DTs = sorted(set(idt).intersection(f._QSArgs.CalcDTRuler))
            Price = Price.reindex(columns=DTs)
        else:
            DTs = Price.columns
        Return = Price.T.pct_change().T
        if f._QSArgs.ModelArgs["mask"]: 
            Mask, x = pd.DataFrame(x[0].T==1, columns=idt, index=SectionIDs), x[1:]
            Mask = (Mask.reindex(columns=DTs).fillna(False) & Price.notnull())
        else:
            Mask = Price.notnull()
        if f._QSArgs.ModelArgs["cat_data"]: 
            CatData, x = pd.DataFrame(x[0].T, columns=idt, index=SectionIDs), x[1:]
            CatData = CatData.reindex(columns=DTs)
        else:
            CatData = None
        if f._QSArgs.ModelArgs["weight"]: 
            Weight, x = pd.DataFrame(x[0].T, columns=idt, index=SectionIDs), x[1:]
            Weight = Weight.reindex(columns=DTs)
        else:
            Weight = pd.DataFrame(1, columns=DTs, index=SectionIDs)
        if CatData is not None:# 进行收益率的类别调整
            Price = Price.where(CatData.notnull(), np.nan)
            AllCates = np.unique(CatData)
            for iCate in AllCates:
                iMask = ((CatData==iCate) & Mask)
                iWeight = Weight.where(iMask, np.nan)
                iReturn = (Return * iWeight.shift(1, axis=1)).sum(axis=0) / iWeight.shift(1, axis=1).sum(axis=0)
                Return = Return.where(~iMask.shift(1, axis=1).fillna(False), Return - iReturn)
        IC, Breadth = pd.DataFrame(index=DTs, columns=iid), pd.DataFrame(index=DTs, columns=iid)
        FactorNames = f._QSArgs.SectionIDs
        Mask = Mask.shift(args["period_lookback"], axis=1).fillna(False)
        for iFactorName in iid:
            if iFactorName not in FactorNames: continue
            iIdx = FactorNames.index(iFactorName)
            iFactorData = pd.DataFrame(x[iIdx].T, columns=idt, index=SectionIDs)
            iFactorData = iFactorData.reindex(columns=DTs).shift(args["period_lookback"], axis=1)
            iMask = (Mask & iFactorData.notnull())
            IC[iFactorName] = Return.where(iMask, np.nan).corrwith(iFactorData, method=args["corr_method"])
            Breadth[iFactorName] = iMask.sum(axis=0)
        Rslt = np.array([IC.reindex(index=idt).values[self._QSArgs.LookBack[0]:], Breadth.reindex(index=idt).values[self._QSArgs.LookBack[0]:]])
        return unstructured_to_structured(Rslt.swapaxes(0, -1), dtype=np.dtype([("IC", float), ("Breadth", float)])).T.astype("O")
        
    def __call__(self, *x:Factor, price:Factor, mask: Optional[Factor]=None, cat_data: Optional[Factor]=None, weight: Optional[Factor]=None, factor_args:dict={}, **kwargs) -> PanelOperation:
        Factors = [price]
        if mask is not None: Factors.append(mask)
        if cat_data is not None: Factors.append(cat_data)
        if weight is not None: Factors.append(weight)
        if not x: raise __QS_Error__("测试因子列表 x 不可为空!")
        else: Factors += x
        if "SectionIDs" not in factor_args:
            PosNum = int(np.log10(len(x))) + 1
            SectionIDs = [f"x-{str(i).zfill(PosNum)}" for i in range(len(x))]
            factor_args["SectionIDs"] = SectionIDs
        elif len(set(factor_args["SectionIDs"]))!=len(x) or (sorted(factor_args["SectionIDs"])!=factor_args["SectionIDs"]):
            raise __QS_Error__(f"截面ID : {factor_args['SectionIDs']} 长度不等于测试因子列表 x 的长度, 或者有重复, 或者非升序排列!")
        factor_args["ModelArgs"] = factor_args.get("ModelArgs", {}) | {"mask": (mask is not None), "cat_data": (cat_data is not None), "weight": (weight is not None), "factor_name_list": [f.Name for f in x]}
        return super().__call__(*Factors, factor_args=factor_args, **kwargs)


class CalcRiskAdjustedIC(PanelOperator):
    """风险调整的 IC 算子"""

    def __init__(self, lookback:int = 31, period_lookback:int=1, corr_method:Literal["spearman", "pearson", "kendall"]="spearman", descriptor_ids:Optional[List[str]]=None, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Arity = args.get("Arity", None) or 1
        Args = {"Name": "calcRiskAdjustedIC"} | args | {"DTMode": "单时点", "OutputMode": "全截面", "DataType": "object"}
        Args["ModelArgs"] = {"corr_method": corr_method, "period_lookback": period_lookback} | Args.get("ModelArgs", {})
        Args["DescriptorSection"] = [Args.get("DescriptorSection", [descriptor_ids])[0]] * Arity
        Args["LookBack"] = [Args.get("LookBack", [lookback])[0]] * Arity
        Args["CompoundType"] = [("IC", float), ("Breadth", float)]
        return super().__init__(args=Args, config_file=config_file, **kwargs)

    def calculate(self, f: Factor, idt: List[dt.datetime], iid: List[str], x: List[np.ndarray], args: dict) -> np.ndarray:
        SectionIDs = (self._QSArgs.DescriptorSection[0] if self._QSArgs.DescriptorSection[0] else iid)
        Price, x = pd.DataFrame(x[0], index=idt, columns=SectionIDs), x[1:]
        if f._QSArgs.CalcDTRuler:
            DTs = sorted(set(idt).intersection(f._QSArgs.CalcDTRuler))
            Price = Price.reindex(index=DTs)
        else:
            DTs = Price.index
        Return = Price.pct_change().iloc[-1]
        if f._QSArgs.ModelArgs["mask"]: 
            Mask, x = pd.DataFrame(x[0]==1, index=idt, columns=SectionIDs), x[1:]
            Mask = (Mask.reindex(index=DTs).fillna(False) & Price.notnull())
        else:
            Mask = Price.notnull()
        Mask = Mask.reindex(index=DTs).shift(args["period_lookback"]).iloc[-1]
        if f._QSArgs.ModelArgs["cat_data"]: 
            CatData, x = pd.DataFrame(x[0], index=idt, columns=SectionIDs), x[1:]
            CatData = CatData.reindex(index=DTs).shift(args["period_lookback"]).iloc[-1]
            _, _, _, DummyFactorData = prepareRegressData(np.ones(CatData.shape[0]), dummy_data=CatData.values)
        else:
            CatData = None
        if f._QSArgs.ModelArgs["risk_factor_list"]:
            nRisk = len(f._QSArgs.ModelArgs["risk_factor_list"])
            RiskExpose, x = pd.DataFrame([pd.DataFrame(ix, index=idt, columns=SectionIDs).reindex(index=DTs).shift(args["period_lookback"]).iloc[-1] for ix in x[:nRisk]]).T, x[nRisk:]
            RiskExpose["constant"] = 1.0
        else:
            RiskExpose = pd.DataFrame(1.0, index=SectionIDs, columns=["constant"])
        iMask = (pd.isnull(RiskExpose).sum(axis=1) == 0)
        iMask = (pd.notnull(Return) & iMask)
        Return = Return[iMask]
        iX = RiskExpose.loc[iMask].values
        if f._QSArgs.ModelArgs["cat_data"]:
            iDummy = DummyFactorData[iMask.values]
            iDummy = iDummy[:, (np.sum(iDummy==0,axis=0)<iDummy.shape[0])]
            iX = np.hstack((iX, iDummy[:,:-1]))
        try:
            Result = sm.OLS(Return.values, iX, missing="drop").fit()
        except:
            return "TODO"
        RiskAdjustedReturn = pd.Series(Result.resid, index=Return.index)
        FactorNames = f._QSArgs.SectionIDs
        FactorExpose = pd.DataFrame([pd.DataFrame(ix, index=idt, columns=SectionIDs).reindex(index=DTs).shift(args["period_lookback"]).iloc[-1] for ix in x]).T
        FactorExpose.columns = FactorNames
        Rslt = np.full(shape=(len(iid), 2), fill_value=np.nan, dtype=float)
        for i, iFactorName in enumerate(iid):
            if iFactorName not in FactorNames: continue
            iFactorExpose = FactorExpose[iFactorName]
            iMask = (Mask & pd.notnull(iFactorExpose))
            iFactorExpose = iFactorExpose[iMask]
            iX = RiskExpose.loc[iMask].values
            if f._QSArgs.ModelArgs["cat_data"]:
                iDummy = DummyFactorData[iMask.values]
                iDummy = iDummy[:, (np.sum(iDummy==0, axis=0) < iDummy.shape[0])]
                iX = np.hstack((iX, iDummy[:,:-1]))
            try:
                Result = sm.OLS(iFactorExpose.values, iX, missing="drop").fit()
            except:
                continue
            iFactorExpose = pd.Series(Result.resid, index=iFactorExpose.index)
            Rslt[i, 0] = iFactorExpose.corr(RiskAdjustedReturn, method=args["corr_method"])
            Rslt[i, 1] = pd.notnull(iFactorExpose).sum()
        return unstructured_to_structured(Rslt).tolist()
        
    def __call__(self, *x:Factor, price:Factor, risk_factors:List[Factor], mask: Optional[Factor]=None, cat_data: Optional[Factor]=None, factor_args:dict={}, **kwargs) -> PanelOperation:
        Factors = [price]
        if mask is not None: Factors.append(mask)
        if cat_data is not None: Factors.append(cat_data)
        Factors += risk_factors
        if not x: raise __QS_Error__("测试因子列表 x 不可为空!")
        else: Factors += x
        if "SectionIDs" not in factor_args:
            PosNum = int(np.log10(len(x))) + 1
            SectionIDs = [f"x-{str(i).zfill(PosNum)}" for i in range(len(x))]
            factor_args["SectionIDs"] = SectionIDs
        elif len(set(factor_args["SectionIDs"]))!=len(x) or (sorted(factor_args["SectionIDs"])!=factor_args["SectionIDs"]):
            raise __QS_Error__(f"截面ID : {factor_args['SectionIDs']} 长度不等于测试因子列表 x 的长度, 或者有重复, 或者非升序排列!")
        factor_args["ModelArgs"] = factor_args.get("ModelArgs", {}) | {"mask": (mask is not None), "cat_data": (cat_data is not None), "factor_name_list": [f.Name for f in x], "risk_factor_list": [f.Name for f in risk_factors]}
        return super().__call__(*Factors, factor_args=factor_args, **kwargs)


class IC(BTNode):
    """IC: 往期因子值和当期收益率的秩相关性。通常采用的相关系数为 Spearman 相关系数。"""
    class __QS_ArgClass__(BTNode.__QS_ArgClass__):
        Name: str = Field(default="IC", frozen=True, title="名称")
        FactorNameList: Optional[List[str]] = Field(default=None, frozen=True, title="因子列表")
        RollingAvgPeriod: int = Field(default=12, frozen=True, title="移动平均期数")
        
    def __init__(self, ic: Factor, args:dict={}, config_file:Optional[str]=None, **kwargs):
        super().__init__(deps=[ic], args=args, config_file=config_file, **kwargs)
    
    def genMatplotlibFig(self, output, file_path=None):
        nRow, nCol = output["IC"].shape[1]//3+(output["IC"].shape[1]%3!=0), min(3, output["IC"].shape[1])
        Fig = Figure(figsize=(min(32, 16+(nCol-1)*8), 8*nRow))
        xData = np.arange(0, output["IC"].shape[0])
        xTicks = np.arange(0, output["IC"].shape[0], max(1, int(output["IC"].shape[0]/10)))
        xTickLabels = [output["IC"].index[i].strftime("%Y-%m-%d") for i in xTicks]
        yMajorFormatter = FuncFormatter(_QS_formatMatplotlibPercentage)
        for i in range(output["IC"].shape[1]):
            iAxes = Fig.add_subplot(nRow, nCol, i+1)
            iAxes.yaxis.set_major_formatter(yMajorFormatter)
            iAxes.plot(xData, output["IC的移动平均"].iloc[:, i].values, label="IC的移动平均", color="indianred", lw=2.5)
            iAxes.bar(xData, output["IC"].iloc[:, i].values, label="IC", color="steelblue")
            iRightAxes = iAxes.twinx()
            iRightAxes.plot(xData, output["截面宽度"].iloc[:, i].values, label="截面宽度", color="k", lw=1.5)
            iAxes.set_xticks(xTicks)
            iAxes.set_xticklabels(xTickLabels)
            iAxes.legend(loc="upper left")
            iRightAxes.legend(loc="upper right")
            iAxes.set_title(output["IC"].columns[i])
        if file_path is not None: Fig.savefig(file_path, dpi=150, bbox_inches='tight')
        return Fig
    
    def genReport(self, output:dict) -> str:
        HTML = "参数设置: "
        HTML += '<ul align="left">'
        if isinstance(getattr(self.Deps[0], "Operator", None), (CalcIC, CalcRiskAdjustedIC)):
            ModelArgs = self.Deps[0].Operator._QSArgs.ModelArgs
            HTML += f"<li>相关性方法: {ModelArgs['corr_method']}</li>"
            HTML += f"<li>回溯期数: {ModelArgs['period_lookback']}</li>"
        if isinstance(getattr(self.Deps[0], "Operator", None), CalcRiskAdjustedIC):
            HTML += f"<li>风险因子: {self.Deps[0]._QSArgs.ModelArgs['risk_factor_list']}</li>"
        if self.Deps[0]._QSArgs.CalcDTRuler:
            HTML += "<li>计算时点: 自定义时点</li>"
        else:
            HTML += "<li>计算时点: 所有时点</li>"
        HTML += f"<li>移动平均期数: {self._QSArgs.RollingAvgPeriod}</li>"
        HTML += "</ul>"
        Formatters = [_QS_formatPandasPercentage]*4+[lambda x:'{0:.4f}'.format(x)]+[lambda x:'{0:.2f}'.format(x)]*3+[lambda x:'{0:.0f}'.format(x)]
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

    def init_compute(self, path: List[str], init_data: BTInitData, context: FactorContext) -> List[FactorInitData]:
        InitData = super().init_compute(path=path, init_data=init_data, context=context)
        return [FactorInitData(DTRange=iInitData.DTRange, SectionIDs=self.Deps[i].getID()) for i, iInitData in enumerate(InitData)]
    
    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[BTLocalContext]=None) -> dict:
        IC, Breadth = bwd_data_list[0].map(lambda x: x[0]), bwd_data_list[0].map(lambda x: x[1])
        if self._QSArgs.FactorNameList:
            FactorNameList = self._QSArgs.FactorNameList
        else:
            FactorNameList = self.Deps[0].Args.ModelArgs.get("factor_name_list", IC.columns)
        IC.columns = Breadth.columns = FactorNameList
        IC = IC.dropna(how="all", axis=0)
        Breadth = Breadth.reindex(index=IC.index)
        Output = {"截面宽度": Breadth, "IC": IC}
        Output["IC的移动平均"] = Output["IC"].copy()
        for i in range(Output["IC"].shape[0]):
            if i<self._QSArgs.RollingAvgPeriod-1: Output["IC的移动平均"].iloc[i, :] = np.nan
            else: Output["IC的移动平均"].iloc[i, :] = Output["IC"].iloc[i-self._QSArgs.RollingAvgPeriod+1:i+1, :].mean()
        Output["统计数据"] = pd.DataFrame(index=Output["IC"].columns)
        Output["统计数据"]["平均值"] = Output["IC"].mean()
        Output["统计数据"]["标准差"] = Output["IC"].std()
        Output["统计数据"]["最小值"] = Output["IC"].min()
        Output["统计数据"]["最大值"] = Output["IC"].max()
        Output["统计数据"]["IC_IR"] = Output["统计数据"]["平均值"] / Output["统计数据"]["标准差"]
        Output["统计数据"]["t统计量"] = np.nan
        Output["统计数据"]["平均截面宽度"] = Output["截面宽度"].mean()
        Output["统计数据"]["IC×Sqrt(N)"] = Output["统计数据"]["平均值"] * np.sqrt(Output["统计数据"]["平均截面宽度"])
        Output["统计数据"]["有效期数"] = 0.0
        for iFactor in Output["IC"]: Output["统计数据"].loc[iFactor, "有效期数"] = pd.notnull(Output["IC"][iFactor]).sum()
        Output["统计数据"]["t统计量"] = Output["统计数据"]["有效期数"]**0.5 * Output["统计数据"]["IC_IR"]
        if self._QSArgs.GenReport: Output["Report"] = self.genReport(Output)
        return Output


class ICDecay(BTNode):
    """IC 衰减: 因子值和收益率关于日期间隔的 IC 衰减情况"""
    class __QS_ArgClass__(BTNode.__QS_ArgClass__):
        Name: str = Field(default="IC 衰减", frozen=True, title="名称")
        PeriodList: Optional[List[str]] = Field(default=None, frozen=True, title="回溯期列表")
        FactorNameList: Optional[List[str]] = Field(default=None, frozen=True, title="因子列表")
        
    def __init__(self, ic_list: List[Factor], args:dict={}, config_file:Optional[str]=None, **kwargs):
        super().__init__(deps=ic_list, args=args, config_file=config_file, **kwargs)
        if self._QSArgs.PeriodList:
            if len(self._QSArgs.PeriodList) != ic_list:
                raise __QS_Error__(f"指定的 PeriodList 的长度 {len(self._QSArgs.PeriodList)} 不等于 IC 因子的数量 {len(ic_list)}")
    
    def genMatplotlibFig(self, output, file_path=None):
        Fig = Figure(figsize=(16, 8))
        xData = np.arange(0, output["统计数据"].shape[0])
        xTickLabels = [str(i) for i in output["统计数据"].index]
        yMajorFormatter = FuncFormatter(_QS_formatMatplotlibPercentage)
        Axes = Fig.add_subplot(1, 1, 1)
        Axes.yaxis.set_major_formatter(yMajorFormatter)
        Axes.bar(xData, output["统计数据"]["IC平均值"].values, label="IC", color="steelblue")
        Axes.set_xticks(xData)
        Axes.set_xticklabels(xTickLabels)
        Axes.legend(loc='upper left')
        RAxes = Axes.twinx()
        RAxes.yaxis.set_major_formatter(yMajorFormatter)
        RAxes.plot(xData, output["统计数据"]["胜率"].values, label="胜率", color="indianred", lw=2.5)
        RAxes.legend(loc="upper right")
        plt.setp(Axes.get_xticklabels(), visible=True, rotation=0, ha='center')
        if file_path is not None: Fig.savefig(file_path, dpi=150, bbox_inches='tight')
        return Fig
    
    def genReport(self, output:dict) -> str:
        HTML = "参数设置: "
        HTML += '<ul align="left">'
        HTML += f"<li>回溯期列表: {self._QSArgs.PeriodList}</li>"
        if self._QSArgs.FactorNameList: HTML += f"<li>因子列表: {self._QSArgs.FactorNameList}</li>"
        HTML += "</ul>"
        Formatters = [_QS_formatPandasPercentage]*2+[lambda x:'{0:.4f}'.format(x), lambda x:'{0:.2f}'.format(x), _QS_formatPandasPercentage]
        for iFactorName in output.keys():
            iHTML = f"因子: {iFactorName}\n"
            iHTML += output[iFactorName]["统计数据"].to_html(formatters=Formatters)
            Pos = iHTML.find(">")
            HTML += iHTML[:Pos]+' align="center"'+iHTML[Pos:]
            Fig = self.genMatplotlibFig(output=output[iFactorName])
            # figure 保存为二进制文件
            Buffer = BytesIO()
            Fig.savefig(Buffer, bbox_inches='tight')
            PlotData = Buffer.getvalue()
            # 图像数据转化为 HTML 格式
            ImgStr = "data:image/png;base64,"+base64.b64encode(PlotData).decode()
            HTML += ('<img src="%s">' % ImgStr)
        return HTML

    def init_compute(self, path: List[str], init_data: BTInitData, context: FactorContext) -> List[FactorInitData]:
        InitData = super().init_compute(path=path, init_data=init_data, context=context)
        return [FactorInitData(DTRange=iInitData.DTRange, SectionIDs=self.Deps[i].getID()) for i, iInitData in enumerate(InitData)]
    
    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[BTLocalContext]=None) -> dict:
        if self._QSArgs.PeriodList:
            PeriodList = self._QSArgs.PeriodList
        else:
            PeriodList = list(range(len(bwd_data_list)))
        IC, Breadth = {}, {}
        FactorNameList, PrefixFactorNameList = None, None
        for i, BwdData in enumerate(bwd_data_list):
            iIC, iBreadth = BwdData.map(lambda x: x[0]), BwdData.map(lambda x: x[1])
            if self._QSArgs.FactorNameList:
                FactorNameList = self._QSArgs.FactorNameList
            else:
                iFactorNameList = self.Deps[i].Args.ModelArgs.get("factor_name_list", iIC.columns.tolist())
                if not FactorNameList:
                    FactorNameList = iFactorNameList
                    if len(set(FactorNameList)) < len(FactorNameList):
                        PrefixFactorNameList = [f"{i}-{iName}" for i, iName in enumerate(FactorNameList)]
                    else:
                        PrefixFactorNameList = FactorNameList
                elif iFactorNameList != FactorNameList: raise __QS_Error__(f"第 {i} 个 IC 因子名列表 '{iFactorNameList}' 和之前的 '{FactorNameList}' 不一致")
            iIC.columns = iBreadth.columns = PrefixFactorNameList
            IC[PeriodList[i]] = iIC
            Breadth[PeriodList[i]] = iBreadth
        IC, Breadth = Panel(IC), Panel(Breadth)
        Output = {}
        for i, iFactorName in enumerate(FactorNameList):
            iIC, iBreadth = IC.iloc[:, :, i], Breadth.iloc[:, :, i]
            iOutput = {"IC": iIC, "Breadth": iBreadth, "统计数据": pd.DataFrame(index=PeriodList)}
            iOutput["统计数据"]["IC平均值"] = iIC.mean()
            nDT = pd.notnull(iIC).sum()
            iOutput["统计数据"]["标准差"] = iIC.std()
            iOutput["统计数据"]["IC_IR"] = iOutput["统计数据"]["IC平均值"] / iOutput["统计数据"]["标准差"]
            iOutput["统计数据"]["t统计量"] = iOutput["统计数据"]["IC_IR"] * nDT ** 0.5
            iOutput["统计数据"]["胜率"] = (iIC > 0).sum() / nDT
            Output[iFactorName] = iOutput
        if self._QSArgs.GenReport: Output["Report"] = self.genReport(Output)
        return Output













        Output = {"截面宽度": Breadth, "IC": IC}
        Output["IC的移动平均"] = Output["IC"].copy()
        for i in range(Output["IC"].shape[0]):
            if i<self._QSArgs.RollingAvgPeriod-1: Output["IC的移动平均"].iloc[i, :] = np.nan
            else: Output["IC的移动平均"].iloc[i, :] = Output["IC"].iloc[i-self._QSArgs.RollingAvgPeriod+1:i+1, :].mean()
        Output["统计数据"] = pd.DataFrame(index=Output["IC"].columns)
        Output["统计数据"]["平均值"] = Output["IC"].mean()
        Output["统计数据"]["标准差"] = Output["IC"].std()
        Output["统计数据"]["最小值"] = Output["IC"].min()
        Output["统计数据"]["最大值"] = Output["IC"].max()
        Output["统计数据"]["IC_IR"] = Output["统计数据"]["平均值"] / Output["统计数据"]["标准差"]
        Output["统计数据"]["t统计量"] = np.nan
        Output["统计数据"]["平均截面宽度"] = Output["截面宽度"].mean()
        Output["统计数据"]["IC×Sqrt(N)"] = Output["统计数据"]["平均值"] * np.sqrt(Output["统计数据"]["平均截面宽度"])
        Output["统计数据"]["有效期数"] = 0.0
        for iFactor in Output["IC"]: Output["统计数据"].loc[iFactor, "有效期数"] = pd.notnull(Output["IC"][iFactor]).sum()
        Output["统计数据"]["t统计量"] = Output["统计数据"]["有效期数"]**0.5 * Output["统计数据"]["IC_IR"]
        if self._QSArgs.GenReport: Output["Report"] = self.genReport(Output)
        return Output
