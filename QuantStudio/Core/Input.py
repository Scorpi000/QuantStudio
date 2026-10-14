# -*- coding: utf-8 -*-
"""输入声明、部分绑定和依赖值结构。

模型字段是输入元信息的唯一来源。部分绑定只检查已经提供的字段，
完整构造使用原模型执行全部校验；节点引用保持共享。
"""
from dataclasses import dataclass
from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin

from annotated_types import MinLen
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, model_validator, create_model
from pydantic_core import PydanticUndefined

from QuantStudio.Core import __QS_Error__


class InputError(__QS_Error__):
    """携带字段路径的输入校验错误。"""

    def __init__(self, diagnostics):
        self.diagnostics = tuple(diagnostics)
        super().__init__("; ".join(f"{'.'.join(map(str, d['path']))}: {d['message']}" for d in diagnostics))


def copy_input(value):
    """复制配置容器，保留计算对象引用。"""
    if isinstance(value, dict):
        return {key: copy_input(item) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_input(item) for item in value]
    if isinstance(value, tuple):
        return tuple(copy_input(item) for item in value)
    return value


def collection_type(annotation):
    """获取集合类型，支持可选的元组字段。"""
    origin = get_origin(annotation)
    if origin is tuple:
        return annotation
    if origin in (Union, UnionType):
        return next((item for item in get_args(annotation) if get_origin(item) is tuple), None)
    return None


class InputModel(BaseModel):
    """以字段声明名称、类型、默认值和数量约束的输入模型。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True, validate_default=True)

    def __reduce__(self):
        return _restore_input_value, (dump_input_model(type(self)), {name: getattr(self, name) for name in type(self).model_fields})

    @classmethod
    def input_info(cls, values=None):
        """返回无需实例化完整模型的输入元信息。

        Args:
            values: 显式绑定值，不包含未指定字段。

        Returns:
            按声明顺序组织的字段元信息字典。
        """
        values = values or {}
        result = {}
        for name, field in cls.model_fields.items():
            result[name] = {
                "name": name, "title": field.title or name,
                "description": field.description or "", "annotation": field.annotation,
                "required": field.is_required(), "collection": collection_type(field.annotation) is not None,
                "min_length": next((getattr(m, "min_length") for m in field.metadata if hasattr(m, "min_length")), None),
                "max_length": next((getattr(m, "max_length") for m in field.metadata if hasattr(m, "max_length")), None),
                "has_default": not field.is_required(),
                "default_factory": field.default_factory,
                "default": None if field.default is PydanticUndefined else copy_input(field.default),
                "bound": name in values,
            }
            if name in values:
                result[name]["value"] = copy_input(values[name])
        return result


class DependencyModel(InputModel):
    """只接受节点、可选节点和节点元组的完整依赖模型。"""

    @model_validator(mode="after")
    def _validate_nodes(self):
        from QuantStudio.Core.Node import Node
        for name in type(self).model_fields:
            value = getattr(self, name)
            items = value if isinstance(value, tuple) else (value,)
            if any(item is not None and not isinstance(item, Node) for item in items):
                raise ValueError(f"依赖 {name} 必须是完整 Node")
        return self


class EmptyDependencies(DependencyModel):
    """不依赖其他计算节点的声明。"""


def dump_input_model(model):
    """记录动态模型的字段定义，避免进程依赖父进程的临时类。"""
    if not getattr(model, "__QS_DynamicInputs__", False):
        return {"type": model}
    return {"name": model.__name__, "dependency": issubclass(model, DependencyModel),
            "fields": {name: (field.annotation, field) for name, field in model.model_fields.items()}}


def restore_input_model(data):
    """在目标进程重建字段定义，静态模型保留其验证器。"""
    if "type" in data:
        return data["type"]
    model = create_model(data["name"], __base__=DependencyModel if data["dependency"] else InputModel,
                         **data["fields"])
    model.__QS_DynamicInputs__ = True
    return model


def _restore_input_value(data, values):
    return restore_input_model(data)(**values)


def _restore_binding(data, values):
    return Binding(restore_input_model(data), values)


@dataclass(frozen=True, init=False)
class Binding:
    """不可变的部分输入绑定。"""

    model: type[BaseModel]
    _items: tuple

    def __reduce__(self):
        return _restore_binding, (dump_input_model(self.model), self.values)

    def __init__(self, model, values=None):
        validated = {}
        for name, value in (values or {}).items():
            if name not in model.model_fields:
                raise InputError([{"code": "unknown_input", "path": (name,), "message": "未声明的输入"}])
            field = model.model_fields[name]
            metadata = tuple(m for m in field.metadata if not isinstance(m, MinLen))
            annotation = Annotated[field.annotation, *metadata] if metadata else field.annotation
            try:
                is_model = isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel)
                config = None if is_model else ConfigDict(arbitrary_types_allowed=True)
                validated[name] = copy_input(TypeAdapter(annotation, config=config).validate_python(value))
            except ValidationError as error:
                raise InputError(self._errors(error, (name,))) from error
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "_items", tuple(validated.items()))

    @staticmethod
    def _errors(error, prefix=()):
        return [{"code": item["type"], "path": prefix + tuple(item["loc"]), "message": item["msg"]}
                for item in error.errors(include_url=False)]

    @property
    def values(self):
        """返回显式绑定的容器副本。"""
        return {name: copy_input(value) for name, value in self._items}

    @property
    def missing(self):
        """尚未提供的必填字段。"""
        values = self.values
        return tuple(name for name, field in self.model.model_fields.items() if field.is_required() and name not in values)

    def bind(self, **values):
        """替换指定槽位并返回新的绑定。"""
        return Binding(self.model, self.values | values)

    def unbind(self, *names):
        """移除指定槽位的显式绑定。"""
        unknown = set(names) - self.model.model_fields.keys()
        if unknown:
            raise InputError([{"code": "unknown_input", "path": (name,), "message": "未声明的输入"} for name in unknown])
        return Binding(self.model, {name: value for name, value in self.values.items() if name not in names})

    def build(self):
        """使用原模型执行完整校验。"""
        try:
            return self.model(**self.values)
        except ValidationError as error:
            raise InputError(self._errors(error)) from error

    def validate(self):
        """返回完整性和模型校验诊断。"""
        try:
            self.build()
        except InputError as error:
            return error.diagnostics
        return ()

    def input_info(self):
        """返回当前声明与显式绑定状态。"""
        return InputModel.input_info.__func__(self.model, self.values)


@dataclass(frozen=True)
class DependencyEdge:
    """包含槽位、集合位置和节点引用的依赖边。"""

    name: str
    index: int | None
    node: Any

    @property
    def path(self):
        return (self.name,) if self.index is None else (self.name, self.index)


def dependency_edges(model):
    """按照字段和集合顺序展开依赖模型。"""
    for name in type(model).model_fields:
        value = getattr(model, name)
        if isinstance(value, tuple):
            for index, node in enumerate(value):
                yield DependencyEdge(name, index, node)
        elif value is not None:
            yield DependencyEdge(name, None, value)


class DependencyValues:
    """按依赖形状组织的只读请求或结果视图。

    属性和字符串索引访问命名字段；整数索引、切片和迭代提供调度及
    数值内核所需的展开视图。字段的结果值不受依赖节点类型约束。
    """

    def __init__(self, dependencies, values):
        object.__setattr__(self, "_dependencies", dependencies)
        if isinstance(values, DependencyValues):
            values = values.flat
        if isinstance(values, dict):
            expected = set(type(dependencies).model_fields)
            if set(values) != expected:
                raise ValueError("依赖值字段必须与依赖模型完全一致")
            flat = []
            for name in type(dependencies).model_fields:
                dependency = getattr(dependencies, name)
                value = values[name]
                if isinstance(dependency, tuple):
                    if not isinstance(value, (tuple, list)) or len(value) != len(dependency):
                        raise ValueError(f"依赖值 {name} 的数量不匹配")
                    flat.extend(value)
                elif dependency is not None:
                    flat.append(value)
                elif value is not None:
                    raise ValueError(f"空依赖 {name} 的结果必须为空")
            values = flat
        values = tuple(values)
        edges = tuple(dependency_edges(dependencies))
        if len(values) != len(edges):
            raise ValueError(f"依赖值数量 {len(values)} 与依赖边数量 {len(edges)} 不一致")
        named = {}
        position = 0
        for name in type(dependencies).model_fields:
            dependency = getattr(dependencies, name)
            if isinstance(dependency, tuple):
                named[name] = values[position:position + len(dependency)]
                position += len(dependency)
            elif dependency is None:
                named[name] = None
            else:
                named[name] = values[position]
                position += 1
        object.__setattr__(self, "_named", named)
        object.__setattr__(self, "_flat", values)

    def __setattr__(self, name, value):
        raise AttributeError("依赖值结构不可修改")

    def __getattr__(self, name):
        try:
            return self._named[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def field(self, name):
        """读取命名结果，支持与视图方法同名的槽位。"""
        return self._named[name]

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._named[key]
        value = self._flat[key]
        return list(value) if isinstance(key, slice) else value

    def __iter__(self):
        return iter(self._flat)

    def __len__(self):
        return len(self._flat)

    def __eq__(self, other):
        if isinstance(other, DependencyValues):
            return self._named == other._named
        if isinstance(other, (tuple, list)):
            return self._flat == tuple(other)
        return NotImplemented

    def __add__(self, other):
        return list(self._flat) + list(other)

    def __radd__(self, other):
        return list(other) + list(self._flat)

    def __reduce__(self):
        return type(self), (self._dependencies, self._flat)

    @property
    def flat(self):
        """按照依赖边顺序组织的结果元组。"""
        return self._flat

    def items(self):
        """按字段顺序读取命名结果。"""
        return self._named.items()
