---
name: python-style
description: Python 项目规范，涵盖开发依赖、数据/模块/注释/异常/日志/函数定义与调用及符号规范。编写或修改 Python 代码、初始化新 Python 项目时使用。
---

# Python 项目规范

新建 Python 项目或编写、修改 Python 代码时遵循本规范。

## 开发依赖

|      工具      |       用途        |
| -------------- | ----------------- |
| pytest         | 同步测试          |
| pytest-asyncio | 异步测试          |
| ruff           | 质量检查 / 格式化 |
| pyright        | 类型检查          |

### ruff 配置

```toml
[tool.ruff]
line-length = 88 # 与 Black 默认值一致
target-version = 'py312' # `[project].requires-python = ">=3.12"` -> `[tool.ruff].target-version = 'py312'`

[tool.ruff.lint]
# 采用全选项 + 单项逐一排除的模式
select = ["ALL"]
# COM812、ISC001 与 formatter 冲突, 且职责已由 formatter(skip-magic-trailing-comma = false)接管
ignore = ["COM812", "ISC001"]

[tool.ruff.lint.pydocstyle]
convention = "google" # 使用 Google 风格的 Docstring

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
line-ending = "auto"
skip-magic-trailing-comma = false
```

### pyright 配置

```toml
[tool.pyright]
typeCheckingMode = "strict"
```

## 模块规范

- **类型导入**：非运行时类型导入统一放在 `TYPE_CHECKING` 中。
- **成员导出**：使用 `__all__` 导出，即使只有一个导出项也要用尾逗号换行。
- **私有成员**：仅在模块内使用的成员以下划线开头命名，如 `def _convert_xxx_to_xxx()`。

## 数据规范

- **外部数据入口**：使用 `pydantic`，保障流入数据安全。
- **内部数据流通**：使用 `dataclass`，以提高性能和节省内存。
- **规范字典结构**：使用 `TypedDict`，方便静态类型检查。

## 注释规范

- **Docstring**：使用 **Google Style Docstring** 格式；公共成员强制添加，私有成员根据复杂度和实际需要添加；语言精练。
- **普通注释**：必要时通过 `NOTE`、`BUG` 等注明类型；语言精练。

## 异常规范

使用 **f-string** + **!r** 格式。

```python
errmsg = f"Invalid value: {value!r}"
raise ValueError(errmsg)
```

## 日志规范

使用 **module logger** + **lazy formatting** + **%r** + **key=value** 格式。

```python
import logging

logger = logging.getLogger(__name__)

logger.debug("Received no message: node=%r, turn=%r", name, turn)
```

## 符号规范

注释、文档字符串与用户可见字符串中统一使用**半角符号**，分隔符后跟**一个空格**；行尾与字符串收尾不留尾随空格。

```python
logger.info("登录成功! 登录态已保存到 %r, 下次可直接使用.")
```

> NOTE: 运行时逻辑中需要匹配全角字符的正则（如页面文本 `IP属地：上海`）不受此规则约束。

## 代码层级划分规范

使用**三级层次注释**划分代码层级，分隔符结束位置与 `pyproject.toml` 中 `[tool.ruff].line-length` 的值相匹配，使用英文时首字母大写。

```python
# ======================================================================================
# Level 1 Title
# ======================================================================================
class BaseXXX:

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Level 2 Title
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _base_convert(self) -> None: ...

    # ----------------------------------------------------------------------------------
    # Level 3 Title
    # ----------------------------------------------------------------------------------
    def convert_a(self) -> None: ...
    def convert_b(self) -> None: ...

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Level 2 Title
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _base_log(self) -> None: ...

    # ----------------------------------------------------------------------------------
    # Level 3 Title
    # ----------------------------------------------------------------------------------
    def log_a(self) -> None: ...
    def log_b(self) -> None: ...
```

## 函数（方法）定义和调用规范

优先使用**关键字参数**，原则上不使用**位置参数**。调用时，如果三方模块强制要求位置参数，遵循即可。定义时，`self`、`cls`、装饰器函数、回调函数、匿名函数等不受此规则约束。

```python
# 函数定义
def handle_1() -> int: ...
def handle_2(*, a: int) -> int: ...
def handle_3(*, a: int, b: int) -> int: ...

# 函数调用
handle_1()
handle_2(a=1)
handle_3(a=1, b=2)

# 方法定义
class H1:
    def handle(self) -> int: ...
class H2:
    def handle(self, *, a: int) -> int: ...
class H3:
    def handle(self, *, a: int, b: int) -> int: ...

# 方法调用
H1().handle()
H2().handle(a=1)
H3().handle(a=1, b=2)
```
