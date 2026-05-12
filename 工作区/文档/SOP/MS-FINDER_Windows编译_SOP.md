# MS-FINDER Windows 编译 SOP

## 概述

MS-FINDER 基于 .NET Framework 4.8.1 开发，需要在 Windows 环境下使用 Visual Studio 或 msbuild 编译。本文档记录完整的编译流程。

## 环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10/11 或 Windows Server |
| .NET Framework | 4.8.1 |
| Visual Studio | 2019 或 2022 |
| msbuild | VS 内置或 Developer Command Prompt |

## 编译步骤

### 方式一：Visual Studio GUI 编译

#### 1. 打开解决方案

```
1. 启动 Visual Studio 2022
2. 文件 → 打开 → 项目/解决方案
3. 选择文件: recetox-msfinder/MsdialWorkbench.sln
```

#### 2. 选择配置

```
1. 解决方案配置: Release
2. 解决方案平台: Any CPU
```

#### 3. 编译项目

```
右键 MsfinderConsoleApp 项目 → 重新生成
```

或使用菜单：`生成 → 重新生成解决方案`

#### 4. 检查输出

```
编译输出目录: MsfinderConsoleApp/bin/Release/
应包含:
├── MsfinderConsoleApp.exe
├── *.dll (依赖库)
└── Resources/ (数据库文件)
```

---

### 方式二：命令行编译 (msbuild)

#### 1. 打开 Developer Command Prompt

**VS 2022:**
```powershell
# 方法1: Windows菜单
开始 → Visual Studio 2022 → x64 Native Tools Command Prompt for VS 2022

# 方法2: 运行命令
C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1
```

#### 2. 进入项目目录

```powershell
cd C:\path\to\recetox-msfinder\MsFinder
```

#### 3. 执行编译

```powershell
msbuild MsfinderConsoleApp.csproj /p:Configuration=Release /p:Platform="Any CPU"
```

或编译整个解决方案:

```powershell
cd C:\path\to\recetox-msfinder
msbuild MsdialWorkbench.sln /p:Configuration=Release /p:Platform="Any CPU"
```

#### 4. 验证编译结果

```powershell
dir MsfinderConsoleApp\bin\Release\*.exe
```

---

## 编译产物

### 目录结构

```
MsFinder/
└── bin/
    └── Release/
        ├── MsfinderConsoleApp.exe      # 主程序
        ├── Common.dll
        ├── Database.dll
        ├── StructureFinder.dll
        ├── MsfinderCommon.dll
        ├── ...
        └── Resources/                   # 数据库文件
            ├── MsfinderFormulaDB-VS11.efd
            ├── MsfinderStructureDB-VS13.esd
            ├── InchikeyClassyfireDB-VS3.icd
            ├── ChemOntologyDB_vs2.cho
            └── ...
```

### 必需文件清单

| 文件类型 | 说明 | 重要性 |
|---------|------|--------|
| `MsfinderConsoleApp.exe` | 主程序 | 必须 |
| `*.dll` | 依赖库 | 必须 |
| `Resources/*.efd` | 分子式数据库 | 必须 |
| `Resources/*.esd` | 结构数据库 | 必须 |
| `Resources/*.icd` | InChIKey分类库 | 必须 |

---

## 部署到 Linux

### 1. 打包编译产物

```powershell
# 在 Windows 上打包
cd C:\path\to\recetox-msfinder\MsFinder\bin\Release
tar -czvf msfinder_release.tar.gz *
```

### 2. 传输到 Linux

```bash
# 使用 scp 或 rsync
scp msfinder_release.tar.gz user@linux:/stor3/AIMS4Meta/源代码/recetox-msfinder/MsFinder/bin/
```

### 3. 在 Linux 上解压

```bash
cd /stor3/AIMS4Meta/源代码/recetox-msfinder/MsFinder/bin/Release
tar -xzvf msfinder_release.tar.gz
```

---

## 配置文件

### MSFINDER.INI

编译产物目录需要包含 `MSFINDER.INI` 配置文件：

```powershell
# 复制 INI 文件
copy C:\path\to\recetox-msfinder\MSFINDER.INI C:\path\to\output\Release\
```

INI 文件内容关键参数：

```ini
# 分子式预测
Ms1Tolerance=0.001
MassToleranceType=ppm

# 结构预测
TreeDepth=2
Ms2Tolerance=0.01

# 数据源
MinesNeverUse=False
MinesOnlyUseForNecessary=True
HMDB=True
UNPD=True
DrugBank=True

# 功能开关
FormulaFinder=True
StructureFinder=True
```

---

## 常见问题

### 1. csproj 缺少 xmlns

**错误:**
```
error MSB4067: The element <Project> beneath element <ProjectCollection> 
is unexpected.
```

**解决:** 确保 csproj 包含 xmlns 属性:

```xml
<Project Sdk="Microsoft.NET.Sdk" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
```

### 2. 缺少 NuGet 包

**解决:**
```powershell
nuget restore MsdialWorkbench.sln
```

### 3. 目标框架不支持

**错误:**
```
error NETSDK1045: The current .NET SDK does not support .NET Framework 4.8.1
```

**解决:** 确保安装 .NET Framework 4.8.1 开发者包，或使用完整 Visual Studio 安装。

---

## 升级编译流程

当官方发布新版本时：

```
1. 下载/同步新版本源码
2. 如有 csproj 变更，参考上方 "常见问题" 修复
3. 执行编译: msbuild MsfinderConsoleApp.csproj /p:Configuration=Release
4. 备份旧版本: mv Release Release_old
5. 部署新版本: cp -r new_release Release
6. 验证运行: mono MsfinderConsoleApp.exe --help
```

---

## 快速检查清单

- [ ] Visual Studio 2019/2022 已安装
- [ ] .NET Framework 4.8.1 SDK 已安装
- [ ] 解决方案能成功打开
- [ ] Release 编译成功，无错误
- [ ] MsfinderConsoleApp.exe 存在于 bin/Release
- [ ] Resources 目录包含数据库文件
- [ ] MSFINDER.INI 已配置
- [ ] Linux 上 mono 能正常运行

---

## 相关文档

- MS-FINDER 源码: `/stor3/AIMS4Meta/源代码/recetox-msfinder/`
- MS-FINDER INI: `/stor3/AIMS4Meta/源代码/recetox-msfinder/MSFINDER.INI`
- 编译产物: `/stor3/AIMS4Meta/源代码/recetox-msfinder/MsFinder/bin/Release/`
