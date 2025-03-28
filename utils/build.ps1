# Copyright 2020 Saleem Abdulrasool <compnerd@compnerd.org>
# Copyright 2023 Tristan Labelle <tristan@thebrowser.company>

<#
.SYNOPSIS
Builds the Swift toolchain, installers, and optionally runs tests.

.DESCRIPTION
This script performs various steps associated with building the Swift toolchain:

- Builds the redistributable, SDK, devtools and toolchain binaries and files
- Builds the msi's and installer executable
- Creates a mock installation under S:\Program Files and S:\Library for local toolchain use
- Optionally runs tests for supported projects
- Optionally stages build artifacts for CI

.PARAMETER SourceCache
The path to a directory where projects contributing to the Swift.
toolchain have been cloned.

.PARAMETER BinaryCache
The path to a directory where to write build system files and outputs.

.PARAMETER ImageRoot
The path to a directory that mimics a file system image root,
under which "Library" and "Program Files" subdirectories will be created
with the files installed by CMake.

.PARAMETER CDebugFormat
The debug information format for C/C++ code: dwarf or codeview.

.PARAMETER SwiftDebugFormat
The debug information format for Swift code: dwarf or codeview.

.PARAMETER AndroidAPILevel
The API Level to target when building the Android SDKs

.PARAMETER Android
When set, build android SDKs.

.PARAMETER AndroidSDKs
An array of architectures for which the Android Swift SDK should be built.

.PARAMETER WindowsSDKs
An array of architectures for which the Windows Swift SDK should be built.

.PARAMETER ProductVersion
The product version to be used when building the installer.
Supports semantic version strings.

.PARAMETER PinnedBuild
The toolchain snapshot to build the early components with.

.PARAMETER PinnedSHA256
The SHA256 for the pinned toolchain.

.PARAMETER PinnedToolchainVariant
The toolchain variant to use while building the toolchain. Defaults to
`Asserts`.

.PARAMETER AndroidNDKVersion
The version number of the Android NDK to be used.

.PARAMETER WinSDKVersion
The version number of the Windows SDK to be used.
Overrides the value resolved by the Visual Studio command prompt.
If no such Windows SDK is installed, it will be downloaded from nuget.

.PARAMETER IncludeDS2
Include the ds2 remote debug server in the SDK.
This component is currently only supported in Android builds.

.PARAMETER SkipBuild
If set, does not run the build phase.

.PARAMETER SkipPackaging
If set, skips building the msi's and installer

.PARAMETER DebugInfo
If set, debug information will be generated for the builds.

.PARAMETER EnableCaching
If true, use `sccache` to cache the build rules.

.PARAMETER Cache
The path to a directory where the `sccache` stores the cache. By default, it will point to `$BinaryCache\sccache`.

.PARAMETER Clean
If true, clean non-compiler builds while building.

.PARAMETER Test
An array of names of projects to run tests for.
'*' runs all tests

.PARAMETER Stage
The path to a directory where built msi's and the installer executable should be staged (for CI).

.PARAMETER BuildTo
The name of a build step after which the script should terminate.
For example: -BuildTo ToolsSupportCore

.PARAMETER ToBatch
When set, runs the script in a special mode which outputs a listing of command invocations
in batch file format instead of executing them.

.PARAMETER HostArchName
The architecture where the toolchain will execute.

.PARAMETER Variant
The toolchain variant to build. Defaults to `Asserts`.

.EXAMPLE
PS> .\Build.ps1

.EXAMPLE
PS> .\Build.ps1 -WindowsSDKs x64 -ProductVersion 1.2.3 -Test foundation,xctest
#>
[CmdletBinding(PositionalBinding = $false)]
param
(
  [System.IO.FileInfo] $SourceCache = "S:\SourceCache",
  [System.IO.FileInfo] $BinaryCache = "S:\b",
  [System.IO.FileInfo] $ImageRoot = "S:",
  [ValidateSet("codeview", "dwarf")]
  [string] $CDebugFormat = "dwarf",
  [ValidateSet("codeview", "dwarf")]
  [string] $SwiftDebugFormat = "dwarf",
  [ValidateRange(1, 36)]
  [int] $AndroidAPILevel = 28,
  [string[]] $AndroidSDKs = @(),
  [string[]] $WindowsSDKs = @("X64","X86","Arm64"),
  [string] $ProductVersion = "0.0.0",
  [string] $ToolchainIdentifier = $(if ($env:TOOLCHAIN_VERSION) { $env:TOOLCHAIN_VERSION } else { "$env:USERNAME.development" }),
  [string] $PinnedBuild = "",
  [ValidatePattern("^[A-Fa-f0-9]{64}$")]
  [string] $PinnedSHA256 = "",
  [string] $PinnedVersion = "",
  [ValidateSet("Asserts", "NoAsserts")]
  [string] $PinnedToolchainVariant = "Asserts",
  [ValidatePattern('^\d+(\.\d+)*$')]
  [string] $PythonVersion = "3.9.10",
  [ValidatePattern("^r(?:[1-9]|[1-9][0-9])(?:[a-z])?$")]
  [string] $AndroidNDKVersion = "r27c",
  [ValidatePattern("^\d+\.\d+\.\d+(?:-\w+)?")]
  [string] $WinSDKVersion = "",
  [switch] $Android = $false,
  [switch] $SkipBuild = $false,
  [switch] $SkipPackaging = $false,
  [switch] $IncludeDS2 = $false,
  [string[]] $Test = @(),
  [string] $Stage = "",
  [ValidateSet("ArgumentParser", "ASN1", "BuildTools", "Certificates", "CMark",
    "Collections", "Compilers", "Crypto", "CURL", "Dispatch", "DocC", "Driver",
    "DS2", "ExperimentalRuntime", "Format", "Foundation", "FoundationMacros",
    "IndexStoreDB", "Inspect", "Installer", "LLBuild", "LLVM", "LMDB",
    "Markdown", "mimalloc", "PackageManager", "PlatformInfoPlist", "RegsGen2",
    "Runtime", "Sanitizers", "SDKSettingsPlist", "SourceKitLSP", "SQLite",
    "System", "Testing", "TestingMacros", "ToolsSupportCore", "XCTest", "XML2",
    "ZLib")]
  [string] $BuildTo = "",
  [ValidateSet("AMD64", "ARM64")]
  [string] $HostArchName = $(if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }),
  [ValidateSet("Asserts", "NoAsserts")]
  [string] $Variant = "Asserts",
  [switch] $Clean,
  [switch] $DebugInfo,
  [switch] $EnableCaching,
  [string] $Cache = "",
  [switch] $Summary,
  [switch] $ToBatch
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

# Avoid being run in a "Developer" shell since this script launches its own sub-shells targeting
# different architectures, and these variables cause confusion.
if ($env:VSCMD_ARG_HOST_ARCH -or $env:VSCMD_ARG_TGT_ARCH) {
  throw "At least one of VSCMD_ARG_HOST_ARCH and VSCMD_ARG_TGT_ARCH is set, which is incompatible with this script. Likely need to run outside of a Developer shell."
}

# Prevent elsewhere-installed swift modules from confusing our builds.
$env:SDKROOT = ""

$BuildArchName = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }

# Validate that if one is set all are set.
if (($PinnedBuild -or $PinnedSHA256 -or $PinnedVersion) -and -not ($PinnedBuild -and $PinnedSHA256 -and $PinnedVersion)) {
  throw "If any of PinnedBuild, PinnedSHA256, or PinnedVersion is set, all three must be set."
}

if (-not $PinnedBuild) {
  switch ($BuildArchName) {
    "AMD64" {
      $PinnedBuild = "https://download.swift.org/swift-6.0.3-release/windows10/swift-6.0.3-RELEASE/swift-6.0.3-RELEASE-windows10.exe"
      $PinnedSHA256 = "AB205D83A38047882DB80E6A88C7D33B651F3BAC96D4515D7CBA5335F37999D3"
      $PinnedVersion = "6.0.3"
    }
    "ARM64" {
      $PinnedBuild = "https://download.swift.org/swift-6.0.3-release/windows10-arm64/swift-6.0.3-RELEASE/swift-6.0.3-RELEASE-windows10-arm64.exe"
      $PinnedSHA256 = "81474651E59A9955C9E6A389EF53ABD61631FFC62C63A2A02977271019E7C722"
      $PinnedVersion = "6.0.3"
    }
    default { throw "Unsupported processor architecture" }
  }
}

$CustomWinSDKRoot = $null # Overwritten if we download a Windows SDK from nuget

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$VSInstallRoot = & $vswhere -nologo -latest -products "*" -all -prerelease -property installationPath
$msbuild = "$VSInstallRoot\MSBuild\Current\Bin\$BuildArchName\MSBuild.exe"

# Avoid $env:ProgramFiles in case this script is running as x86
$UnixToolsBinDir = "$env:SystemDrive\Program Files\Git\usr\bin"

if ($Android -and ($AndroidSDKs.Length -eq 0)) {
  # Enable all android SDKs by default.
  $AndroidSDKs = @("aarch64","armv7","i686","x86_64")
}
# Work around limitations of cmd passing in array arguments via powershell.exe -File
if ($AndroidSDKs.Length -eq 1) { $AndroidSDKs = $AndroidSDKs[0].Split(",") }
if ($WindowsSDKs.Length -eq 1) { $WindowsSDKs = $WindowsSDKs[0].Split(",") }
if ($Test.Length -eq 1) { $Test = $Test[0].Split(",") }

if ($AndroidSDKs.Length -gt 0) {
  # Always enable android when one of the SDKs is specified.
  $Android = $true
}

if ($Test -contains "*") {
  # Explicitly don't include llbuild yet since tests are known to fail on Windows
  $Test = @("lld", "lldb", "swift", "dispatch", "foundation", "xctest", "swift-format", "sourcekit-lsp")
}

# Architecture definitions
$ArchX64 = @{
  VSName = "amd64";
  ShortName = "x64";
  LLVMName = "x86_64";
  LLVMTarget = "x86_64-unknown-windows-msvc";
  CMakeName = "AMD64";
  BinaryDir = "bin64";
  ToolchainInstallRoot = "$BinaryCache\x64\toolchains\$ProductVersion+$Variant";
  Cache = @{};
}

$ArchX86 = @{
  VSName = "x86";
  ShortName = "x86";
  LLVMName = "i686";
  LLVMTarget = "i686-unknown-windows-msvc";
  CMakeName = "i686";
  BinaryDir = "bin32";
  Cache = @{};
}

$ArchARM64 = @{
  VSName = "arm64";
  ShortName = "arm64";
  LLVMName = "aarch64";
  LLVMTarget = "aarch64-unknown-windows-msvc";
  CMakeName = "ARM64";
  BinaryDir = "bin64a";
  ToolchainInstallRoot = "$BinaryCache\arm64\toolchains\$ProductVersion+$Variant";
  Cache = @{};
}

$AndroidARM64 = @{
  AndroidArchABI = "arm64-v8a";
  BinaryDir = "bin64a";
  CMakeName = "aarch64";
  LLVMName = "aarch64";
  LLVMTarget = "aarch64-unknown-linux-android$AndroidAPILevel";
  ShortName = "arm64";
  Cache = @{};
}

$AndroidARMv7 = @{
  AndroidArchABI = "armeabi-v7a";
  BinaryDir = "bina";
  CMakeName = "armv7-a";
  LLVMName = "armv7";
  LLVMTarget = "armv7-unknown-linux-androideabi$AndroidAPILevel";
  ShortName = "armv7";
  Cache = @{};
}

$AndroidX86 = @{
  AndroidArchABI = "x86";
  BinaryDir = "bin";
  CMakeName = "i686";
  LLVMName = "i686";
  LLVMTarget = "i686-unknown-linux-android$AndroidAPILevel";
  ShortName = "x86";
  Cache = @{};
}

$AndroidX64 = @{
  AndroidArchABI = "x86_64";
  BinaryDir = "bin64";
  CMakeName = "x86_64";
  LLVMName = "x86_64";
  LLVMTarget = "x86_64-unknown-linux-android$AndroidAPILevel";
  ShortName = "x64";
  Cache = @{};
}

$HostArch = switch ($HostArchName) {
  "AMD64" { $ArchX64 }
  "ARM64" { $ArchARM64 }
  default { throw "Unsupported processor architecture" }
}

$BuildArch = switch ($BuildArchName) {
  "AMD64" { $ArchX64 }
  "ARM64" { $ArchARM64 }
  default { throw "Unsupported processor architecture" }
}

$WiX = @{
  Version = "4.0.6";
  URL = "https://www.nuget.org/api/v2/package/wix/4.0.6";
  SHA256 = "A94DD42AE1FB56B32DA180E2173CEDA4F0D10B4C8871C5EE59ECB502131A1EB6";
  Path = [IO.Path]::Combine("$BinaryCache\WiX-4.0.6", "tools", "net6.0", "any");
}

$KnownPythons = @{
  "3.9.10" = @{
    AMD64 = @{
      URL = "https://www.nuget.org/api/v2/package/python/3.9.10";
      SHA256 = "ac43b491e9488ac926ed31c5594f0c9409a21ecbaf99dc7a93f8c7b24cf85867";
    };
    ARM64 = @{
      URL = "https://www.nuget.org/api/v2/package/pythonarm64/3.9.10";
      SHA256 = "429ada77e7f30e4bd8ff22953a1f35f98b2728e84c9b1d006712561785641f69";
    };
  }
}

$PythonWheels = @{
  "packaging" = @{
    File = "packaging-24.1-py3-none-any.whl";
    URL = "https://files.pythonhosted.org/packages/08/aa/cc0199a5f0ad350994d660967a8efb233fe0416e4639146c089643407ce6/packaging-24.1-py3-none-any.whl";
    SHA256 = "5b8f2217dbdbd2f7f384c41c628544e6d52f2d0f53c6d0c3ea61aa5d1d7ff124";
  };
  "distutils" = @{
    File = "setuptools-75.1.0-py3-none-any.whl";
    URL = "https://files.pythonhosted.org/packages/ff/ae/f19306b5a221f6a436d8f2238d5b80925004093fa3edea59835b514d9057/setuptools-75.1.0-py3-none-any.whl";
    SHA256 = "35ab7fd3bcd95e6b7fd704e4a1539513edad446c097797f2985e0e4b960772f2";
  };
  "psutil" = @{
    File = "psutil-6.1.0-cp37-abi3-win_amd64.whl";
    URL = "https://files.pythonhosted.org/packages/11/91/87fa6f060e649b1e1a7b19a4f5869709fbf750b7c8c262ee776ec32f3028/psutil-6.1.0-cp37-abi3-win_amd64.whl";
    SHA256 = "a8fb3752b491d246034fa4d279ff076501588ce8cbcdbb62c32fd7a377d996be";
  };
  "unittest2" = @{
    File = "unittest2-1.1.0-py2.py3-none-any.whl";
    URL = "https://files.pythonhosted.org/packages/72/20/7f0f433060a962200b7272b8c12ba90ef5b903e218174301d0abfd523813/unittest2-1.1.0-py2.py3-none-any.whl";
    SHA256 = "13f77d0875db6d9b435e1d4f41e74ad4cc2eb6e1d5c824996092b3430f088bb8";
  };
}

$KnownNDKs = @{
  r26b = @{
    URL = "https://dl.google.com/android/repository/android-ndk-r26b-windows.zip"
    SHA256 = "A478D43D4A45D0D345CDA6BE50D79642B92FB175868D9DC0DFC86181D80F691E"
    ClangVersion = 17
  }
  r27c = @{
    URL = "https://dl.google.com/android/repository/android-ndk-r27c-windows.zip"
    SHA256 = "27E49F11E0CEE5800983D8AF8F4ACD5BF09987AA6F790D4439DDA9F3643D2494"
    ClangVersion = 18
  }
}

$IsCrossCompiling = $HostArchName -ne $BuildArchName

$TimingData = New-Object System.Collections.Generic.List[System.Object]

function Add-TimingData {
  param
  (
    [Parameter(Mandatory)]
    [string] $Arch,
    [Parameter(Mandatory)]
    [Platform] $Platform,
    [Parameter(Mandatory)]
    [string] $BuildStep,
    [Parameter(Mandatory)]
    [System.TimeSpan] $ElapsedTime
  )

  $TimingData.Add([PSCustomObject]@{
    Arch = $Arch
    Platform = $Platform
    "Build Step" = $BuildStep
    "Elapsed Time" = $ElapsedTime
  })
}

function Write-Summary {
  Write-Host "Summary:" -ForegroundColor Cyan

  $TotalTime = [TimeSpan]::Zero
  foreach ($Entry in $TimingData) {
    $TotalTime = $TotalTime.Add($Entry."Elapsed Time")
  }

  $SortedData = $TimingData | ForEach-Object {
    $Percentage = [math]::Round(($_.("Elapsed Time").TotalSeconds / $TotalTime.TotalSeconds) * 100, 1)
    $FormattedTime = "{0:hh\:mm\:ss\.ff}" -f $_."Elapsed Time"
    [PSCustomObject]@{
      "Build Step" = $_."Build Step"
      Platform = $_.Platform
      Arch = $_.Arch
      "Elapsed Time" = $FormattedTime
      "%" = "$Percentage%"
    }
  } | Sort-Object -Descending -Property "%"

  $FormattedTotalTime = "{0:hh\:mm\:ss\.ff}" -f $TotalTime
  $TotalRow = [PSCustomObject]@{
    "Build Step" = "TOTAL"
    Platform = ""
    Arch = ""
    "Elapsed Time" = $FormattedTotalTime
    "%" = "100.0%"
  }

  @($SortedData) + $TotalRow | Format-Table -AutoSize
}

function Get-AndroidNDK {
  $NDK = $KnownNDKs[$AndroidNDKVersion]
  if (-not $NDK) { throw "Unsupported Android NDK version" }
  return $NDK
}

function Get-AndroidNDKPath {
  return Join-Path -Path $BinaryCache -ChildPath "android-ndk-$AndroidNDKVersion"
}

function Get-FlexExecutable {
  return Join-Path -Path $BinaryCache -ChildPath "win_flex_bison\win_flex.exe"
}

function Get-BisonExecutable {
  return Join-Path -Path $BinaryCache -ChildPath "win_flex_bison\win_bison.exe"
}

function Get-PythonExecutable {
  return Join-Path -Path $BinaryCache -ChildPath "Python$($BuildArch.CMakeName)-$PythonVersion\tools\python.exe"
}

function Get-InstallDir($Arch) {
  if ($Arch -eq $HostArch) {
    $ProgramFilesName = "Program Files"
  } elseif ($Arch -eq $ArchX86) {
    $ProgramFilesName = "Program Files (x86)"
  } elseif (($HostArch -eq $ArchArm64) -and ($Arch -eq $ArchX64)) {
    # x64 programs actually install under "Program Files" on arm64,
    # but this would conflict with the native installation.
    $ProgramFilesName = "Program Files (Amd64)"
  } else {
    # arm64 cannot be installed on x64
    return $null
  }
  return "$ImageRoot\$ProgramFilesName\Swift"
}

$NugetRoot = "$BinaryCache\nuget"
$PinnedToolchain = [IO.Path]::GetFileNameWithoutExtension($PinnedBuild)

$LibraryRoot = "$ImageRoot\Library"

# For dev productivity, install the host toolchain directly using CMake.
# This allows iterating on the toolchain using ninja builds.
$HostArch.ToolchainInstallRoot = "$(Get-InstallDir $HostArch)\Toolchains\$ProductVersion+$Variant"

# Resolve the architectures received as argument
$AndroidSDKArchs = @($AndroidSDKs | ForEach-Object {
  switch ($_) {
    "aarch64" { $AndroidARM64 }
    "armv7" { $AndroidARMv7 }
    "i686" { $AndroidX86 }
    "x86_64" { $AndroidX64 }
    default { throw "Unknown architecture $_" }
  }
})
if ($Android) {
  if ($HostArch -ne $ArchX64) {
    throw "Unsupported host architecture for building android SDKs"
  }
}
$WindowsSDKArchs = @($WindowsSDKs | ForEach-Object {
  switch ($_) {
    "X64" { $ArchX64 }
    "X86" { $ArchX86 }
    "Arm64" { $ArchArm64 }
    default { throw "Unknown architecture $_" }
  }
})

# Build functions
function Invoke-BuildStep([string] $Name) {
  if ($Summary) {
    # TODO: Replace hacky introspection of $Args with Platform object.
    $BuildStepPlatform = "Windows"
    $BuildStepArch = "(n/a)"
    $Stopwatch = [Diagnostics.Stopwatch]::StartNew()
    foreach ($Arg in $Args) {
      switch ($Arg.GetType().Name) {
        "Hashtable" {
          if ($Arg.ContainsKey("LLVMName")) { $BuildStepArch = $Arg["LLVMName"] }
        }
        "string" {
          if ($Arg -eq "Windows" -or $Arg -eq "Android") { $BuildStepPlatform = $Arg }
        }
      }
    }
  }

  & $Name @Args

  if ($Summary) {
    Add-TimingData $BuildStepArch $BuildStepPlatform ($Name -replace "Build-","") $Stopwatch.Elapsed
  }
  if ($Name.Replace("Build-", "") -eq $BuildTo) {
    exit 0
  }
}

enum TargetComponent {
  LLVM
  Runtime
  Dispatch
  DynamicFoundation
  XCTest
  Testing
  ClangBuiltins
  ClangRuntime
  SwiftInspect
  ExperimentalRuntime
  StaticFoundation
}

function Get-TargetProjectBinaryCache($Arch, [TargetComponent]$Project) {
  return "$BinaryCache\$($Arch.LLVMTarget)\$Project"
}

function Get-TargetProjectCMakeModules($Arch, [TargetComponent]$Project) {
  return "$Binarycache\$($Arch.LLVMTarget)\$Project\cmake\modules"
}

enum HostComponent {
  Compilers
  FoundationMacros
  TestingMacros
  ToolsSupportCore
  LLBuild
  ArgumentParser
  Driver
  Crypto
  Collections
  ASN1
  Certificates
  System
  Build
  PackageManager
  Markdown
  Format
  LMDB
  IndexStoreDB
  SourceKitLSP
  SymbolKit
  DocC
}

function Get-HostProjectBinaryCache([HostComponent]$Project) {
  if ($Project -eq "Compilers") { return "$BinaryCache\5" }
  return "$BinaryCache\$($HostArch.LLVMTarget)\$Project"
}

function Get-HostProjectCMakeModules([HostComponent]$Project) {
  if ($Project -eq "Compilers") { return "$BinaryCache\5\cmake\modules" }
  return "$BinaryCache\$($HostArch.LLVMTarget)\$Project\cmake\modules"
}

enum BuildComponent {
  BuildTools
  Compilers
  FoundationMacros
  TestingMacros
  RegsGen2
}

function Get-BuildProjectBinaryCache([BuildComponent]$Project) {
  if ($Project -eq "Compilers") { return "$BinaryCache\1" }
  return "$BinaryCache\$($BuildArch.LLVMTarget)\$Project"
}

function Get-BuildProjectCMakeModules([BuildComponent]$Project) {
  if ($Project -eq "Compilers") { return "$BinaryCache\1\cmake\modules" }
  return "$BinaryCache\$($BuildArch.LLVMTarget)\$Project\cmake\modules"
}

function Get-TargetInfo($Arch) {
  # Cache the result of "swift -print-target-info" as $Arch.Cache.TargetInfo
  $CacheKey = "TargetInfo"
  if (-not $Arch.Cache.ContainsKey($CacheKey)) {
    Invoke-IsolatingEnvVars {
      $env:Path = "$(Get-PinnedToolchainRuntime);$(Get-PinnedToolchainToolsDir);${env:Path}"
      $TargetInfo = & swiftc -target $Arch.LLVMTarget -print-target-info
      if ($LastExitCode -ne 0) {
        throw "Unable to print target info for '$($Arch.LLVMTarget)'"
      }
      $TargetInfo = $TargetInfo | ConvertFrom-JSON
      $Arch.Cache[$CacheKey] = $TargetInfo.target
    }
  }
  return $Arch.Cache[$CacheKey]
}

function Get-ModuleTriple($Arch) {
  return (Get-TargetInfo -Arch $Arch).moduleTriple
}

function Copy-File($Src, $Dst) {
  # Create the directory tree first so Copy-Item succeeds
  # If $Dst is the target directory, make sure it ends with "\"
  $DstDir = [IO.Path]::GetDirectoryName($Dst)
  if ($ToBatch) {
    Write-Output "md `"$DstDir`""
    Write-Output "copy /Y `"$Src`" `"$Dst`""
  } else {
    New-Item -ItemType Directory -ErrorAction Ignore $DstDir | Out-Null
    Copy-Item -Force $Src $Dst
  }
}

function Copy-Directory($Src, $Dst) {
  if ($ToBatch) {
    Write-Output "md `"$Dst`""
    Write-Output "copy /Y `"$Src`" `"$Dst`""
  } else {
    New-Item -ItemType Directory -ErrorAction Ignore $Dst | Out-Null
    Copy-Item -Force -Recurse $Src $Dst
  }
}

function Move-Directory($Src, $Dst) {
  if ($ToBatch) {
  } else {
    $Destination = Join-Path -Path $Dst -ChildPath (Split-Path -Path $Src -Leaf)
    if (Test-Path -Path $Destination -Type Container) {
      Remove-Item -Path $Destination -Recurse -Force | Out-Null
    }
    Move-Item -Path $Src -Destination $Dst -Force | Out-Null
  }
}

function Invoke-Program() {
  [CmdletBinding(PositionalBinding = $false)]
  param
  (
    [Parameter(Position = 0, Mandatory = $true)]
    [string] $Executable,
    [switch] $Silent,
    [switch] $OutNull,
    [string] $OutFile = "",
    [string] $ErrorFile = "",
    [Parameter(Position = 1, ValueFromRemainingArguments)]
    [string[]] $ExecutableArgs
  )

  if ($ToBatch) {
    # Print the invocation in batch file-compatible format
    $OutputLine = "`"$Executable`""
    $ShouldBreakLine = $false
    for ($i = 0; $i -lt $ExecutableArgs.Length; $i++) {
      if ($ShouldBreakLine -or $OutputLine.Length -ge 40) {
        $OutputLine += " ^"
        Write-Output $OutputLine
        $OutputLine = "  "
      }

      $Arg = $ExecutableArgs[$i]
      if ($Arg.Contains(" ")) {
        $OutputLine += " `"$Arg`""
      } else {
        $OutputLine += " $Arg"
      }

      # Break lines after non-switch arguments
      $ShouldBreakLine = -not $Arg.StartsWith("-")
    }

    if ($OutNull) {
      $OutputLine += " > nul"
    } elseif ($Silent) {
      $OutputLine += " *> nul"
    } else {
      if ($OutFile) { $OutputLine += " > `"$OutFile`"" }
      if ($ErrorFile) { $OutputLine += " 2> `"$ErrorFile`"" }
    }

    Write-Output $OutputLine
  } else {
    if ($OutNull) {
      & $Executable @ExecutableArgs | Out-Null
    } elseif ($Silent) {
      & $Executable @ExecutableArgs *> $null
    } elseif ($OutFile -and $ErrorFile) {
      & $Executable @ExecutableArgs > $OutFile 2> $ErrorFile
    } elseif ($OutFile) {
      & $Executable @ExecutableArgs > $OutFile
    } elseif ($ErrorFile) {
      & $Executable @ExecutableArgs 2> $ErrorFile
    } else {
      & $Executable @ExecutableArgs
    }

    if ($LastExitCode -ne 0) {
      $ErrorMessage = "Error: $([IO.Path]::GetFileName($Executable)) exited with code $($LastExitCode).`n"

      $ErrorMessage += "Invocation:`n"
      $ErrorMessage += "  $Executable $ExecutableArgs`n"

      $ErrorMessage += "Call stack:`n"
      foreach ($Frame in @(Get-PSCallStack)) {
        $ErrorMessage += "  $Frame`n"
      }

      throw $ErrorMessage
    }
  }
}

function Invoke-IsolatingEnvVars([scriptblock]$Block) {
  if ($ToBatch) {
    Write-Output "setlocal enableextensions enabledelayedexpansion"
  }

  $OldVars = @{}
  foreach ($Var in (Get-ChildItem env:*).GetEnumerator()) {
    $OldVars.Add($Var.Key, $Var.Value)
  }

  & $Block

  Remove-Item env:*
  foreach ($Var in $OldVars.GetEnumerator()) {
    New-Item -Path "env:\$($Var.Key)" -Value $Var.Value -ErrorAction Ignore | Out-Null
  }

  if ($ToBatch) {
    Write-Output "endlocal"
  }
}

function Invoke-VsDevShell($Arch) {
  $DevCmdArguments = "-no_logo -host_arch=$($BuildArch.VSName) -arch=$($Arch.VSName)"
  if ($CustomWinSDKRoot) {
    $DevCmdArguments += " -winsdk=none"
  } elseif ($WinSDKVersion) {
    $DevCmdArguments += " -winsdk=$WinSDKVersion"
  }

  if ($ToBatch) {
    Write-Output "call `"$VSInstallRoot\Common7\Tools\VsDevCmd.bat`" $DevCmdArguments"
  } else {
    # This dll path is valid for VS2019 and VS2022, but it was under a vsdevcmd subfolder in VS2017
    Import-Module "$VSInstallRoot\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
    Enter-VsDevShell -VsInstallPath $VSInstallRoot -SkipAutomaticLocation -DevCmdArguments $DevCmdArguments

    if ($CustomWinSDKRoot) {
      # Using a non-installed Windows SDK. Setup environment variables manually.
      $WinSDKVerIncludeRoot = "$CustomWinSDKRoot\include\$WinSDKVersion"
      $WinSDKIncludePath = "$WinSDKVerIncludeRoot\ucrt;$WinSDKVerIncludeRoot\um;$WinSDKVerIncludeRoot\shared;$WinSDKVerIncludeRoot\winrt;$WinSDKVerIncludeRoot\cppwinrt"
      $WinSDKVerLibRoot = "$CustomWinSDKRoot\lib\$WinSDKVersion"

      $env:WindowsLibPath = "$CustomWinSDKRoot\UnionMetadata\$WinSDKVersion;$CustomWinSDKRoot\References\$WinSDKVersion"
      $env:WindowsSdkBinPath = "$CustomWinSDKRoot\bin"
      $env:WindowsSDKLibVersion = "$WinSDKVersion\"
      $env:WindowsSdkVerBinPath = "$CustomWinSDKRoot\bin\$WinSDKVersion"
      $env:WindowsSDKVersion = "$WinSDKVersion\"

      $env:EXTERNAL_INCLUDE += ";$WinSDKIncludePath"
      $env:INCLUDE += ";$WinSDKIncludePath"
      $env:LIB += ";$WinSDKVerLibRoot\ucrt\$($Arch.ShortName);$WinSDKVerLibRoot\um\$($Arch.ShortName)"
      $env:LIBPATH += ";$env:WindowsLibPath"
      $env:PATH += ";$env:WindowsSdkVerBinPath\$($Arch.ShortName);$env:WindowsSdkBinPath\$($Arch.ShortName)"
      $env:UCRTVersion = $WinSDKVersion
      $env:UniversalCRTSdkDir = $CustomWinSDKRoot
    }
  }
}

function Get-Dependencies {
  Write-Host "[$([DateTime]::Now.ToString("yyyy-MM-dd HH:mm:ss"))] Fetch-Dependencies ..." -ForegroundColor Cyan
  $ProgressPreference = "SilentlyContinue"

  $WebClient = New-Object Net.WebClient

  function DownloadAndVerify($URL, $Destination, $Hash) {
    if (Test-Path $Destination) {
      return
    }

    Write-Output "$Destination not found. Downloading ..."
    if ($ToBatch) {
      Write-Output "md `"$(Split-Path -Path $Destination -Parent)`""
      Write-Output "curl.exe -sL $URL -o $Destination"
      Write-Output "(certutil -HashFile $Destination SHA256) == $Hash || (exit /b)"
    } else {
      New-Item -ItemType Directory (Split-Path -Path $Destination -Parent) -ErrorAction Ignore | Out-Null
      $WebClient.DownloadFile($URL, $Destination)
      $SHA256 = Get-FileHash -Path $Destination -Algorithm SHA256
      if ($SHA256.Hash -ne $Hash) {
        throw "SHA256 mismatch ($($SHA256.Hash) vs $Hash)"
      }
    }
  }

  function Expand-ZipFile {
    param
    (
        [string]$ZipFileName,
        [string]$BinaryCache,
        [string]$ExtractPath,
        [bool]$CreateExtractPath = $true
    )

    $source = Join-Path -Path $BinaryCache -ChildPath $ZipFileName
    $destination = Join-Path -Path $BinaryCache -ChildPath $ExtractPath

    # Check if the extracted directory already exists and is up to date.
    if (Test-Path $destination) {
        $zipLastWriteTime = (Get-Item $source).LastWriteTime
        $extractedLastWriteTime = (Get-Item $destination).LastWriteTime
        # Compare the last write times
        if ($zipLastWriteTime -le $extractedLastWriteTime) {
            Write-Output "'$ZipFileName' is already extracted and up to date."
            return
        }
    }

    $destination = if ($CreateExtractPath) { $destination } else { $BinaryCache }

    Write-Output "Extracting '$ZipFileName' ..."
    New-Item -ItemType Directory -ErrorAction Ignore -Path $BinaryCache | Out-Null
    Expand-Archive -Path $source -DestinationPath $destination -Force
  }

  function Export-Toolchain {
    param
    (
        [string]$InstallerExeName,
        [string]$BinaryCache,
        [string]$ToolchainName
    )

    $source = Join-Path -Path $BinaryCache -ChildPath $InstallerExeName
    $destination = Join-Path -Path $BinaryCache -ChildPath toolchains\$ToolchainName

    # Check if the extracted directory already exists and is up to date.
    if (Test-Path $destination) {
        $installerWriteTime = (Get-Item $source).LastWriteTime
        $extractedWriteTime = (Get-Item $destination).LastWriteTime
        if ($installerWriteTime -le $extractedWriteTime) {
            Write-Output "'$InstallerExeName' is already extracted and up to date."
            return
        }
    }

    Write-Output "Extracting '$InstallerExeName' ..."

    # The new runtime MSI is built to expand files into the immediate directory. So, setup the installation location.
    New-Item -ItemType Directory -ErrorAction Ignore $BinaryCache\toolchains\$PinnedToolchain\LocalApp\Programs\Swift\Runtimes\$(Get-PinnedToolchainVersion)\usr\bin | Out-Null
    Invoke-Program "$($WiX.Path)\wix.exe" -- burn extract $BinaryCache\$InstallerExeName -out $BinaryCache\toolchains\ -outba $BinaryCache\toolchains\
    Get-ChildItem "$BinaryCache\toolchains\WixAttachedContainer" -Filter "*.msi" | ForEach-Object {
      $LogFile = [System.IO.Path]::ChangeExtension($_.Name, "log")
      $TARGETDIR = if ($_.Name -eq "rtl.msi") { "$BinaryCache\toolchains\$ToolchainName\LocalApp\Programs\Swift\Runtimes\$(Get-PinnedToolchainVersion)\usr\bin" } else { "$BinaryCache\toolchains\$ToolchainName" }
      Invoke-Program -OutNull msiexec.exe /lvx! $BinaryCache\toolchains\$LogFile /qn /a $BinaryCache\toolchains\WixAttachedContainer\$($_.Name) ALLUSERS=0 TARGETDIR=$TARGETDIR
    }
  }

  if ($SkipBuild -and $SkipPackaging) { return }

  $Stopwatch = [Diagnostics.Stopwatch]::StartNew()
  if ($ToBatch) {
    Write-Host -ForegroundColor Cyan "[$([DateTime]::Now.ToString("yyyy-MM-dd HH:mm:ss"))] Get-Dependencies..."
  }

  DownloadAndVerify $WiX.URL "$BinaryCache\WiX-$($WiX.Version).zip" $WiX.SHA256
  Expand-ZipFile WiX-$($WiX.Version).zip $BinaryCache WiX-$($WiX.Version)

  if ($SkipBuild) { return }

  DownloadAndVerify $PinnedBuild "$BinaryCache\$PinnedToolchain.exe" $PinnedSHA256

  if ($Test -contains "lldb") {
    # The make tool isn't part of MSYS
    $GnuWin32MakeURL = "https://downloads.sourceforge.net/project/ezwinports/make-4.4.1-without-guile-w32-bin.zip"
    $GnuWin32MakeHash = "fb66a02b530f7466f6222ce53c0b602c5288e601547a034e4156a512dd895ee7"
    DownloadAndVerify $GnuWin32MakeURL "$BinaryCache\GnuWin32Make-4.4.1.zip" $GnuWin32MakeHash
    Expand-ZipFile GnuWin32Make-4.4.1.zip $BinaryCache GnuWin32Make-4.4.1
  }

  # TODO(compnerd) stamp/validate that we need to re-extract
  New-Item -ItemType Directory -ErrorAction Ignore $BinaryCache\toolchains | Out-Null
  Export-Toolchain "$PinnedToolchain.exe" $BinaryCache $PinnedToolchain

  function Get-KnownPython([string] $ArchName) {
    if (-not $KnownPythons.ContainsKey($PythonVersion)) {
      throw "Unknown python version: $PythonVersion"
    }
    return $KnownPythons[$PythonVersion].$ArchName
  }

  function Install-Python([string] $ArchName) {
    $Python = Get-KnownPython $ArchName
    DownloadAndVerify $Python.URL "$BinaryCache\Python$ArchName-$PythonVersion.zip" $Python.SHA256
    if (-not $ToBatch) {
      Expand-ZipFile Python$ArchName-$PythonVersion.zip "$BinaryCache" Python$ArchName-$PythonVersion
    }
  }

  function Install-PIPIfNeeded() {
    try {
      Invoke-Program -Silent "$(Get-PythonExecutable)" -m pip
    } catch {
      Write-Output "Installing pip ..."
      Invoke-Program -OutNull "$(Get-PythonExecutable)" '-I' -m ensurepip -U --default-pip
    } finally {
      Write-Output "pip installed."
    }
  }

  function Install-PythonWheel([string] $ModuleName) {
    try {
      Invoke-Program -Silent "$(Get-PythonExecutable)" -c "import $ModuleName"
    } catch {
      $Wheel = $PythonWheels[$ModuleName]
      DownloadAndVerify $Wheel.URL "$BinaryCache\python\$($Wheel.File)" $Wheel.SHA256
      Write-Output "Installing '$($Wheel.File)' ..."
      Invoke-Program -OutNull "$(Get-PythonExecutable)" '-I' -m pip install "$BinaryCache\python\$($Wheel.File)" --disable-pip-version-check
    } finally {
      Write-Output "$ModuleName installed."
    }
  }

  function Install-PythonModules() {
    Install-PIPIfNeeded
    Install-PythonWheel "packaging" # For building LLVM 18+
    Install-PythonWheel "distutils" # Required for SWIG support
    if ($Test -contains "lldb") {
      Install-PythonWheel "psutil" # Required for testing LLDB
      Install-PythonWheel "unittest2" # Required for testing LLDB
    }
  }

  Install-Python $HostArchName
  if ($IsCrossCompiling) {
    Install-Python $BuildArchName
  }
  # Ensure Python modules that are required as host build tools
  Install-PythonModules

  if ($Android) {
    $NDK = Get-AndroidNDK
    DownloadAndVerify $NDK.URL "$BinaryCache\android-ndk-$AndroidNDKVersion-windows.zip" $NDK.SHA256
    Expand-ZipFile -ZipFileName "android-ndk-$AndroidNDKVersion-windows.zip" -BinaryCache $BinaryCache -ExtractPath "android-ndk-$AndroidNDKVersion" -CreateExtractPath $false
  }

  if ($IncludeDS2) {
    $WinFlexBisonVersion = "2.5.25"
    $WinFlexBisonURL = "https://github.com/lexxmark/winflexbison/releases/download/v$WinFlexBisonVersion/win_flex_bison-$WinFlexBisonVersion.zip"
    $WinFlexBisonHash = "8D324B62BE33604B2C45AD1DD34AB93D722534448F55A16CA7292DE32B6AC135"
    DownloadAndVerify $WinFlexBisonURL "$BinaryCache\win_flex_bison-$WinFlexBisonVersion.zip" $WinFlexBisonHash

    Expand-ZipFile -ZipFileName "win_flex_bison-$WinFlexBisonVersion.zip" -BinaryCache $BinaryCache -ExtractPath "win_flex_bison"
  }

  if ($WinSDKVersion) {
    try {
      # Check whether VsDevShell can already resolve the requested Windows SDK Version
      Invoke-IsolatingEnvVars { Invoke-VsDevShell $HostArch }
    } catch {
      $Package = Microsoft.Windows.SDK.CPP

      Write-Output "Windows SDK $WinSDKVersion not found. Downloading from nuget.org ..."
      Invoke-Program nuget install $Package -Version $WinSDKVersion -OutputDirectory $NugetRoot

      # Set to script scope so Invoke-VsDevShell can read it.
      $script:CustomWinSDKRoot = "$NugetRoot\$Package.$WinSDKVersion\c"

      # Install each required architecture package and move files under the base /lib directory.
      $WinSDKArchs = $WindowsSDKArchs.Clone()
      if (-not ($HostArch -in $WinSDKArchs)) {
        $WinSDKArch += $HostArch
      }

      foreach ($Arch in $WinSDKArchs) {
        Invoke-Program nuget install $Package.$($Arch.ShortName) -Version $WinSDKVersion -OutputDirectory $NugetRoot
        Copy-Directory "$NugetRoot\$Package.$($Arch.ShortName).$WinSDKVersion\c\*" "$CustomWinSDKRoot\lib\$WinSDKVersion"
      }
    }
  }

  if (-not $ToBatch) {
    Write-Host -ForegroundColor Cyan "[$([DateTime]::Now.ToString("yyyy-MM-dd HH:mm:ss"))] Get-Dependencies took $($Stopwatch.Elapsed)"
    Write-Host ""
  }
  if ($Summary) {
    Add-TimingData $BuildArch.LLVMName "Windows" "Get-Dependencies" $Stopwatch.Elapsed
  }
}

function Get-PinnedToolchainToolsDir() {
  $ToolchainsRoot = [IO.Path]::Combine("$BinaryCache\toolchains", "$PinnedToolchain", "LocalApp", "Programs", "Swift", "Toolchains")

  # NOTE: Add a workaround for the main snapshots that inadvertently used the
  # wrong version when they were built. This allows use of the nightly snapshot
  # as a pinned toolchain.
  if ((Get-PinnedToolchainVersion) -eq "0.0.0") {
    if (-not (Test-Path "$ToolchainsRoot\0.0.0+Asserts\usr\bin")) {
      if (Test-Path "$ToolchainsRoot\6.0.0+Asserts\usr\bin") {
        return "$ToolchainsRoot\6.0.0+Asserts\usr\bin"
      }
    }
  }

  $VariantToolchainPath = [IO.Path]::Combine($ToolchainsRoot, "$(Get-PinnedToolchainVersion)+$PinnedToolchainVariant", "usr", "bin")

  if (Test-Path $VariantToolchainPath) {
    return $VariantToolchainPath
  }

  return "$BinaryCache\toolchains\${PinnedToolchain}\Library\Developer\Toolchains\unknown-$PinnedToolchainVariant-development.xctoolchain\usr\bin"
}

function Get-PinnedToolchainSDK() {
  if (Test-Path "$BinaryCache\toolchains\${PinnedToolchain}\LocalApp\Programs\Swift\Platforms\$(Get-PinnedToolchainVersion)\Windows.platform\Developer\SDKs\Windows.sdk") {
    return "$BinaryCache\toolchains\${PinnedToolchain}\LocalApp\Programs\Swift\Platforms\$(Get-PinnedToolchainVersion)\Windows.platform\Developer\SDKs\Windows.sdk"
  }
  return "$BinaryCache\toolchains\${PinnedToolchain}\Library\Developer\Platforms\Windows.platform\Developer\SDKs\Windows.sdk"
}

function Get-PinnedToolchainRuntime() {
  if (Test-Path "$BinaryCache\toolchains\${PinnedToolchain}\LocalApp\Programs\Swift\Runtimes\$(Get-PinnedToolchainVersion)\usr\bin\swiftCore.dll") {
    return "$BinaryCache\toolchains\${PinnedToolchain}\LocalApp\Programs\Swift\Runtimes\$(Get-PinnedToolchainVersion)\usr\bin"
  }
  return "$BinaryCache\toolchains\${PinnedToolchain}\PFiles64\Swift\runtime-development\usr\bin"
}

function Get-PinnedToolchainVersion() {
  if (Test-Path variable:PinnedVersion) {
    return $PinnedVersion
  }
  throw "PinnedVersion must be set"
}

function Add-KeyValueIfNew([hashtable]$Hashtable, [string]$Key, [string]$Value) {
  if (-not $Hashtable.Contains($Key)) {
    $Hashtable.Add($Key, $Value)
  }
}

function Add-FlagsDefine([hashtable]$Defines, [string]$Name, [string[]]$Value) {
  if ($Defines.Contains($Name)) {
    $Defines[$name] = @($Defines[$name]) + $Value
  } else {
    $Defines.Add($Name, $Value)
  }
}

function Test-SCCacheAtLeast([int]$Major, [int]$Minor, [int]$Patch = 0) {
  if ($ToBatch) { return $false }

  $SCCacheVersionString = @(& sccache.exe --version)[0]
  if (-not ($SCCacheVersionString -match "sccache (\d+)\.(\d+)(?:\.(\d+))?")) {
    throw "Unexpected SCCache version string format"
  }

  if ([int]$Matches.1 -ne $Major) { return [int]$Matches.1 -gt $Major }
  if ([int]$Matches.2 -ne $Minor) { return [int]$Matches.2 -gt $Minor }
  if ($null -eq $Matches.3) { return 0 -gt $Patch }
  return [int]$Matches.3 -ge $Patch
}

enum Platform {
  Windows
  Android
}

function Get-PlatformRoot([Platform] $Platform) {
  return ([IO.Path]::Combine((Get-InstallDir $HostArch), "Platforms", "${Platform}.platform"))
}

function Get-SwiftSDK {
  [CmdletBinding(PositionalBinding = $false)]
  param
  (
    [Parameter(Position = 0, Mandatory = $true)]
    [Platform] $Platform,
    [string] $Identifier = $Platform.ToString()
  )
  return ([IO.Path]::Combine((Get-PlatformRoot $Platform), "Developer", "SDKs", "${Identifier}.sdk"))
}

function Build-CMakeProject {
  [CmdletBinding(PositionalBinding = $false)]
  param(
    [string] $Src,
    [string] $Bin,
    [string] $InstallTo = "",
    [Platform] $Platform = "Windows",
    [hashtable] $Arch,
    [string] $Generator = "Ninja",
    [string] $CacheScript = "",
    [string[]] $UseMSVCCompilers = @(), # C,CXX
    [string[]] $UseBuiltCompilers = @(), # ASM,C,CXX,Swift
    [string[]] $UsePinnedCompilers = @(), # ASM,C,CXX,Swift
    [switch] $UseSwiftSwiftDriver = $false,
    [switch] $AddAndroidCMakeEnv = $false,
    [switch] $UseGNUDriver = $false,
    [string] $SwiftSDK = "",
    [hashtable] $Defines = @{}, # Values are either single strings or arrays of flags
    [string[]] $BuildTargets = @()
  )

  if ($ToBatch) {
    Write-Output ""
    Write-Output "echo Building '$Src' to '$Bin' ..."
  } else {
    Write-Host -ForegroundColor Cyan "[$([DateTime]::Now.ToString("yyyy-MM-dd HH:mm:ss"))] Building '$Src' to '$Bin' ..."
  }

  $Stopwatch = [Diagnostics.Stopwatch]::StartNew()

  # Enter the developer command shell early so we can resolve cmake.exe
  # for version checks.
  Invoke-IsolatingEnvVars {
    if ($Platform -eq "Windows") {
      Invoke-VsDevShell $Arch
    }

    $CompilersBinaryCache = if ($IsCrossCompiling) {
      Get-BuildProjectBinaryCache Compilers
    } else {
      Get-HostProjectBinaryCache Compilers
    }
    $DriverBinaryCache = Get-HostProjectBinaryCache Driver

    if ($EnableCaching) {
      $env:SCCACHE_DIRECT = "true"
      if ($Cache -eq "") {
        $env:SCCACHE_DIR = "$BinaryCache\sccache"
      } else {
        $env:SCCACHE_DIR = $Cache
      }
    }
    if ($UseSwiftSwiftDriver) {
      $env:SWIFT_DRIVER_SWIFT_FRONTEND_EXEC = ([IO.Path]::Combine($CompilersBinaryCache, "bin", "swift-frontend.exe"))
    }

    # TODO(compnerd) workaround swiftc.exe symlink not existing.
    if ($UseSwiftSwiftDriver) {
      Copy-Item -Force ([IO.Path]::Combine($DriverBinaryCache, "bin", "swift-driver.exe")) ([IO.Path]::Combine($DriverBinaryCache, "bin", "swiftc.exe"))
    }

    # Add additional defines (unless already present)
    $Defines = $Defines.Clone()

    if (($Platform -ne "Windows") -or ($Arch.CMakeName -ne $BuildArch.CMakeName)) {
      Add-KeyValueIfNew $Defines CMAKE_SYSTEM_NAME $Platform
      Add-KeyValueIfNew $Defines CMAKE_SYSTEM_PROCESSOR $Arch.CMakeName
    }

    if ($AddAndroidCMakeEnv) {
      # Set generic android options if we need to build an Android runtime component
      # while building the compiler. Use an environment variable to pass it, to
      # ensure that it can be accessed from the cmake cache file.
      $env:NDKPATH = Get-AndroidNDKPath
    }
    if ($Platform -eq "Android") {
      $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
      $VSInstallPath = & $vswhere -nologo -latest -products * -property installationPath
      if (Test-Path "${VSInstallPath}\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin") {
        $env:Path = "${VSInstallPath}\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;${VSInstallPath}\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;${env:Path}"
        Add-KeyValueIfNew $Defines CMAKE_MAKE_PROGRAM "${VSInstallPath}\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
      } else {
        throw "Missing CMake and Ninja in the visual studio installation that are needed to build Android"
      }
      $androidNDKPath = Get-AndroidNDKPath
      Add-KeyValueIfNew $Defines CMAKE_C_COMPILER (Join-Path -Path $androidNDKPath -ChildPath "toolchains\llvm\prebuilt\windows-x86_64\bin\clang.exe")
      Add-KeyValueIfNew $Defines CMAKE_CXX_COMPILER (Join-Path -Path $androidNDKPath -ChildPath "toolchains\llvm\prebuilt\windows-x86_64\bin\clang++.exe")
      Add-KeyValueIfNew $Defines CMAKE_ANDROID_API "$AndroidAPILevel"
      Add-KeyValueIfNew $Defines CMAKE_ANDROID_ARCH_ABI $Arch.AndroidArchABI
      Add-KeyValueIfNew $Defines CMAKE_ANDROID_NDK "$androidNDKPath"
      Add-KeyValueIfNew $Defines SWIFT_ANDROID_NDK_PATH "$androidNDKPath"
      Add-KeyValueIfNew $Defines CMAKE_C_COMPILER_WORKS YES
      Add-KeyValueIfNew $Defines CMAKE_CXX_COMPILER_WORKS YES
    }

    Add-KeyValueIfNew $Defines CMAKE_BUILD_TYPE Release

    $CFlags = @()
    switch ($Platform) {
      Windows {
        $CFlags = if ($UseGNUDriver) {
          @("-fno-stack-protector", "-ffunction-sections", "-fdata-sections", "-fomit-frame-pointer")
        } else {
          @("/GS-", "/Gw", "/Gy", "/Oi", "/Oy", "/Zc:inline")
        }
      }
      Android {
        $CFlags = @("--sysroot=$(Get-AndroidNDKPath)\toolchains\llvm\prebuilt\windows-x86_64\sysroot")
      }
    }

    $CXXFlags = @()
    if ($Platform -eq "Windows" -and -not $UseGNUDriver) {
      $CXXFlags += $CFlags.Clone() + @("/Zc:__cplusplus")
    }

    if ($UseMSVCCompilers.Contains("C") -Or $UseMSVCCompilers.Contains("CXX") -Or
        $UseBuiltCompilers.Contains("C") -Or $UseBuiltCompilers.Contains("CXX") -Or
        $UsePinnedCompilers.Contains("C") -Or $UsePinnedCompilers.Contains("CXX")) {
      if ($DebugInfo -and $Platform -eq "Windows") {
        Add-FlagsDefine $Defines CMAKE_MSVC_DEBUG_INFORMATION_FORMAT Embedded
        Add-FlagsDefine $Defines CMAKE_POLICY_CMP0141 NEW
        # Add additional linker flags for generating the debug info.
        if ($UseGNUDriver) {
          Add-FlagsDefine $Defines CMAKE_SHARED_LINKER_FLAGS @("-Xlinker", "-debug")
          Add-FlagsDefine $Defines CMAKE_EXE_LINKER_FLAGS @("-Xlinker", "-debug")
        } else {
          Add-FlagsDefine $Defines CMAKE_SHARED_LINKER_FLAGS "/debug"
          Add-FlagsDefine $Defines CMAKE_EXE_LINKER_FLAGS "/debug"
        }
      } elseif ($Platform -eq "Android") {
        # Use a built lld linker as the Android's NDK linker might be too
        # old and not support all required relocations needed by the Swift
        # runtime.
        $ldPath = ([IO.Path]::Combine($CompilersBinaryCache, "bin", "ld.lld"))
        Add-FlagsDefine $Defines CMAKE_SHARED_LINKER_FLAGS "--ld-path=$ldPath"
        Add-FlagsDefine $Defines CMAKE_EXE_LINKER_FLAGS "--ld-path=$ldPath"
      }
    }

    if ($UseMSVCCompilers.Contains("C")) {
      Add-KeyValueIfNew $Defines CMAKE_C_COMPILER cl
      if ($EnableCaching) {
        Add-KeyValueIfNew $Defines CMAKE_C_COMPILER_LAUNCHER sccache
      }
      Add-FlagsDefine $Defines CMAKE_C_FLAGS $CFlags
    }
    if ($UseMSVCCompilers.Contains("CXX")) {
      Add-KeyValueIfNew $Defines CMAKE_CXX_COMPILER cl
      if ($EnableCaching) {
        Add-KeyValueIfNew $Defines CMAKE_CXX_COMPILER_LAUNCHER sccache
      }
      Add-FlagsDefine $Defines CMAKE_CXX_FLAGS $CXXFlags
    }
    if ($UsePinnedCompilers.Contains("ASM") -Or $UseBuiltCompilers.Contains("ASM")) {
      $Driver = if ($Platform -eq "Windows") { "clang-cl.exe" } else { "clang.exe" }
      if ($UseBuiltCompilers.Contains("ASM")) {
        Add-KeyValueIfNew $Defines CMAKE_ASM_COMPILER ([IO.Path]::Combine($CompilersBinaryCache, "bin", $Driver))
      } else {
        Add-KeyValueIfNew $Defines CMAKE_ASM_COMPILER (Join-Path -Path (Get-PinnedToolchainToolsDir) -ChildPath $Driver)
      }
      Add-FlagsDefine $Defines CMAKE_ASM_FLAGS "--target=$($Arch.LLVMTarget)"
      if ($Platform -eq "Windows") {
        Add-KeyValueIfNew $Defines CMAKE_ASM_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDLL "/MD"
      }
    }
    if ($UsePinnedCompilers.Contains("C") -Or $UseBuiltCompilers.Contains("C")) {
      $Driver = if ($Platform -eq "Windows" -and -not $UseGNUDriver) { "clang-cl.exe" } else { "clang.exe" }
      if ($UseBuiltCompilers.Contains("C")) {
        Add-KeyValueIfNew $Defines CMAKE_C_COMPILER ([IO.Path]::Combine($CompilersBinaryCache, "bin", $Driver))
      } else {
        Add-KeyValueIfNew $Defines CMAKE_C_COMPILER (Join-Path -Path (Get-PinnedToolchainToolsDir) -ChildPath $Driver)
      }
      Add-KeyValueIfNew $Defines CMAKE_C_COMPILER_TARGET $Arch.LLVMTarget

      if ($DebugInfo -and $CDebugFormat -eq "dwarf") {
        Add-FlagsDefine $Defines CMAKE_C_FLAGS "-gdwarf"
      }
      Add-FlagsDefine $Defines CMAKE_C_FLAGS $CFlags
    }
    if ($UsePinnedCompilers.Contains("CXX") -Or $UseBuiltCompilers.Contains("CXX")) {
      $Driver = if ($Platform -eq "Windows" -and -not $UseGNUDriver) { "clang-cl.exe" } else { "clang++.exe" }
      if ($UseBuiltCompilers.Contains("CXX")) {
        Add-KeyValueIfNew $Defines CMAKE_CXX_COMPILER ([IO.Path]::Combine($CompilersBinaryCache, "bin", $Driver))
      } else {
        Add-KeyValueIfNew $Defines CMAKE_CXX_COMPILER (Join-Path -Path (Get-PinnedToolchainToolsDir) -ChildPath $Driver)
      }
      Add-KeyValueIfNew $Defines CMAKE_CXX_COMPILER_TARGET $Arch.LLVMTarget

      if ($DebugInfo -and $CDebugFormat -eq "dwarf") {
        Add-FlagsDefine $Defines CMAKE_CXX_FLAGS "-gdwarf"
      }
      Add-FlagsDefine $Defines CMAKE_CXX_FLAGS $CXXFlags
    }
    if ($UsePinnedCompilers.Contains("Swift") -Or $UseBuiltCompilers.Contains("Swift")) {
      $SwiftArgs = @()

      if ($UseSwiftSwiftDriver) {
        Add-KeyValueIfNew $Defines CMAKE_Swift_COMPILER ([IO.Path]::Combine($DriverBinaryCache, "bin", "swiftc.exe"))
      } elseif ($UseBuiltCompilers.Contains("Swift")) {
        Add-KeyValueIfNew $Defines CMAKE_Swift_COMPILER ([IO.Path]::Combine($CompilersBinaryCache, "bin", "swiftc.exe"))
      } else {
        Add-KeyValueIfNew $Defines CMAKE_Swift_COMPILER (Join-Path -Path (Get-PinnedToolchainToolsDir) -ChildPath  "swiftc.exe")
      }
      if (-not ($Platform -eq "Windows")) {
        Add-KeyValueIfNew $Defines CMAKE_Swift_COMPILER_WORKS = "YES"
      }
      if ($UseBuiltCompilers.Contains("Swift")) {
        Add-KeyValueIfNew $Defines CMAKE_Swift_COMPILER_TARGET (Get-ModuleTriple $Arch)
        $RuntimeBinaryCache = Get-TargetProjectBinaryCache $Arch Runtime
        $SwiftResourceDir = "${RuntimeBinaryCache}\lib\swift"

        switch ($Platform) {
          Windows {
            if ($SwiftSDK -ne "") {
              $SwiftArgs += @("-sdk", $SwiftSDK)
            } else {
              $SwiftArgs += @(
                "-vfsoverlay", "$RuntimeBinaryCache\stdlib\windows-vfs-overlay.yaml",
                "-strict-implicit-module-context",
                "-Xcc", "-Xclang", "-Xcc", "-fbuiltin-headers-in-system-modules"
              )
              $SwiftArgs += @("-resource-dir", "$SwiftResourceDir")
              $SwiftArgs += @("-L", "$SwiftResourceDir\$($Platform.ToString().ToLowerInvariant())")
            }
          }
          Android {
            $androidNDKPath = Get-AndroidNDKPath
            if ($SwiftSDK -ne "") {
              $SwiftArgs += @("-sdk", $SwiftSDK)
              $SwiftArgs += @("-sysroot", "$androidNDKPath\toolchains\llvm\prebuilt\windows-x86_64\sysroot")
            } else {
              $SwiftArgs += @("-sdk", "$androidNDKPath\toolchains\llvm\prebuilt\windows-x86_64\sysroot")
              $SwiftArgs += @("-resource-dir", "$SwiftResourceDir")
              $SwiftArgs += @("-L", "$SwiftResourceDir\$($Platform.ToString().ToLowerInvariant())")
            }
            $SwiftArgs += @(
              "-Xclang-linker", "-target",
              "-Xclang-linker", $Arch.LLVMTarget,
              "-Xclang-linker", "--sysroot",
              "-Xclang-linker", "$androidNDKPath\toolchains\llvm\prebuilt\windows-x86_64\sysroot",
              "-Xclang-linker", "-resource-dir",
              "-Xclang-linker", "$androidNDKPath\toolchains\llvm\prebuilt\windows-x86_64\lib\clang\$($(Get-AndroidNDK).ClangVersion)"
            )
          }
        }

      } else {
        Add-KeyValueIfNew $Defines CMAKE_Swift_COMPILER_TARGET $Arch.LLVMTarget
        $SwiftArgs += @("-sdk", (Get-PinnedToolchainSDK))
      }

      # Debug Information
      if ($DebugInfo) {
        if ($Platform -eq "Windows") {
          if ($SwiftDebugFormat -eq "dwarf") {
            $SwiftArgs += @("-g", "-Xlinker", "/DEBUG:DWARF", "-use-ld=lld-link")
          } else {
            $SwiftArgs += @("-g", "-debug-info-format=codeview", "-Xlinker", "-debug")
          }
        } else {
          $SwiftArgs += @("-g")
        }
      } else {
        $SwiftArgs += "-gnone"
      }

      if ($Platform -eq "Windows") {
        $SwiftArgs += @("-Xlinker", "/INCREMENTAL:NO")
        # Swift requires COMDAT folding and de-duplication
        $SwiftArgs += @("-Xlinker", "/OPT:REF")
        $SwiftArgs += @("-Xlinker", "/OPT:ICF")
      }

      Add-FlagsDefine $Defines CMAKE_Swift_FLAGS $SwiftArgs

      # Workaround CMake 3.26+ enabling `-wmo` by default on release builds
      Add-FlagsDefine $Defines CMAKE_Swift_FLAGS_RELEASE "-O"
      Add-FlagsDefine $Defines CMAKE_Swift_FLAGS_RELWITHDEBINFO "-O"
    }
    if ("" -ne $InstallTo) {
      Add-KeyValueIfNew $Defines CMAKE_INSTALL_PREFIX $InstallTo
    }

    # Generate the project
    $cmakeGenerateArgs = @("-B", $Bin, "-S", $Src, "-G", $Generator)
    if ("" -ne $CacheScript) {
      $cmakeGenerateArgs += @("-C", $CacheScript)
    }
    foreach ($Define in ($Defines.GetEnumerator() | Sort-Object Name)) {
      # The quoting gets tricky to support defines containing compiler flags args,
      # some of which can contain spaces, for example `-D` `Flags=-flag "C:/Program Files"`
      # Avoid backslashes since they are going into CMakeCache.txt,
      # where they are interpreted as escapes.
      if ($Define.Value -is [string]) {
        # Single token value, no need to quote spaces, the splat operator does the right thing.
        $Value = $Define.Value.Replace("\", "/")
      } else {
        # Flags array, multiple tokens, quoting needed for tokens containing spaces
        $Value = ""
        foreach ($Arg in $Define.Value) {
          if ($Value.Length -gt 0) {
            $Value += " "
          }

          $ArgWithForwardSlashes = $Arg.Replace("\", "/")
          if ($ArgWithForwardSlashes.Contains(" ")) {
            # Quote and escape the quote so it makes it through
            $Value += "\""$ArgWithForwardSlashes\"""
          } else {
            $Value += $ArgWithForwardSlashes
          }
        }
      }

      $cmakeGenerateArgs += @("-D", "$($Define.Key)=$Value")
    }

    if ($UseBuiltCompilers.Contains("Swift")) {
      $env:Path = "$([IO.Path]::Combine((Get-InstallDir $BuildArch), "Runtimes", $ProductVersion, "usr", "bin"));$(Get-CMarkBinaryCache $BuildArch)\src;$($BuildArch.ToolchainInstallRoot)\usr\bin;$(Get-PinnedToolchainRuntime);${env:Path}"
    } elseif ($UsePinnedCompilers.Contains("Swift")) {
      $env:Path = "$(Get-PinnedToolchainRuntime);${env:Path}"
    }

    if ($ToBatch) {
      Write-Output ""
      Write-Output "echo cmake.exe $cmakeGenerateArgs"
    } else {
      Write-Host "cmake.exe $cmakeGenerateArgs"
    }
    Invoke-Program cmake.exe @cmakeGenerateArgs

    # Build all requested targets
    foreach ($Target in $BuildTargets) {
      if ($Target -eq "default") {
        Invoke-Program cmake.exe --build $Bin
      } else {
        Invoke-Program cmake.exe --build $Bin --target $Target
      }
    }

    if ($BuildTargets.Length -eq 0 -and $InstallTo) {
      Invoke-Program cmake.exe --build $Bin --target install
    }
  }

  if (-not $ToBatch) {
    Write-Host -ForegroundColor Cyan "[$([DateTime]::Now.ToString("yyyy-MM-dd HH:mm:ss"))] Finished building '$Src' to '$Bin' in $($Stopwatch.Elapsed)"
    Write-Host ""
  }
}

enum SPMBuildAction {
  # 'swift build'
  Build
  # 'swift test'
  Test
  # 'swift test --parallel'
  TestParallel
}

function Build-SPMProject {
  [CmdletBinding(PositionalBinding = $false)]
  param(
    [SPMBuildAction] $Action,
    [string] $Src,
    [string] $Bin,
    [hashtable] $Arch,
    [Parameter(ValueFromRemainingArguments)]
    [string[]] $AdditionalArguments
  )

  $ActionForOutput = switch ($Action) {
    Build { "Building" }
    Test { "Testing" }
    TestParallel { "Testing" }
  }

  if ($ToBatch) {
    Write-Output ""
    Write-Output "echo $ActionForOutput '$Src' to '$Bin' ..."
  } else {
    Write-Host -ForegroundColor Cyan "[$([DateTime]::Now.ToString("yyyy-MM-dd HH:mm:ss"))] $ActionForOutput '$Src' to '$Bin' ..."
  }

  $Stopwatch = [Diagnostics.Stopwatch]::StartNew()

  Invoke-IsolatingEnvVars {
    $RuntimeInstallRoot = [IO.Path]::Combine((Get-InstallDir $HostArch), "Runtimes", $ProductVersion)

    $env:Path = "$RuntimeInstallRoot\usr\bin;$($HostArch.ToolchainInstallRoot)\usr\bin;${env:Path}"
    $env:SDKROOT = (Get-SwiftSDK Windows)
    $env:SWIFTCI_USE_LOCAL_DEPS = "1"

    $Arguments = @(
        "--scratch-path", $Bin,
        "--package-path", $Src,
        "-c", "release",
        "-Xbuild-tools-swiftc", "-I$(Get-SwiftSDK Windows)\usr\lib\swift",
        "-Xbuild-tools-swiftc", "-L$(Get-SwiftSDK Windows)\usr\lib\swift\windows",
        "-Xcc", "-I$(Get-SwiftSDK Windows)\usr\lib\swift",
        "-Xlinker", "-L$(Get-SwiftSDK Windows)\usr\lib\swift\windows"
    )
    if ($DebugInfo) {
      if ($SwiftDebugFormat -eq "dwarf") {
        $Arguments += @("-debug-info-format", "dwarf")
      } else {
        $Arguments += @("-debug-info-format", "codeview")
      }
    } else {
      $Arguments += @("-debug-info-format", "none")
    }

    switch ($Action) {
      Build {
        $ActionName = "build"
      }
      Test {
        $ActionName = "test"
      }
      TestParallel {
        $ActionName = "test"
        $Arguments += @("--parallel")
      }
    }

    Invoke-Program "$($HostArch.ToolchainInstallRoot)\usr\bin\swift.exe" $ActionName @Arguments @AdditionalArguments
  }

  if (-not $ToBatch) {
    Write-Host -ForegroundColor Cyan "[$([DateTime]::Now.ToString("yyyy-MM-dd HH:mm:ss"))] Finished building '$Src' to '$Bin' in $($Stopwatch.Elapsed)"
    Write-Host ""
  }

  if ($Summary) {
    Add-TimingData $BuildArch.LLVMName "Windows" $Src.Replace($SourceCache, '') $Stopwatch.Elapsed
  }
}

function Build-WiXProject() {
  [CmdletBinding(PositionalBinding = $false)]
  param
  (
    [Parameter(Position = 0, Mandatory = $true)]
    [string]$FileName,
    [Parameter(Mandatory = $true)]
    [hashtable]$Arch,
    [switch]$Bundle,
    [hashtable]$Properties = @{}
  )

  $ProductVersionArg = $ProductVersion
  if (-not $Bundle) {
    # WiX v4 will accept a semantic version string for Bundles,
    # but Packages still require a purely numerical version number,
    # so trim any semantic versioning suffixes
    $ProductVersionArg = [regex]::Replace($ProductVersion, "[-+].*", "")
  }

  $Properties = $Properties.Clone()
  Add-KeyValueIfNew $Properties Configuration Release
  Add-KeyValueIfNew $Properties BaseOutputPath "$BinaryCache\$($Arch.LLVMTarget)\installer\"
  Add-KeyValueIfNew $Properties ProductArchitecture $Arch.VSName
  Add-KeyValueIfNew $Properties ProductVersion $ProductVersionArg

  $MSBuildArgs = @("$SourceCache\swift-installer-scripts\platforms\Windows\$FileName")
  $MSBuildArgs += "-noLogo"
  $MSBuildArgs += "-restore"
  $MSBuildArgs += "-maxCpuCount"
  foreach ($Property in $Properties.GetEnumerator()) {
    if ($Property.Value.Contains(" ")) {
      $MSBuildArgs += "-p:$($Property.Key)=$($Property.Value.Replace('\', '\\'))"
    } else {
      $MSBuildArgs += "-p:$($Property.Key)=$($Property.Value)"
    }
  }
  $MSBuildArgs += "-binaryLogger:$BinaryCache\$($Arch.LLVMTarget)\msi\$($Arch.VSName)-$([System.IO.Path]::GetFileNameWithoutExtension($FileName)).binlog"
  $MSBuildArgs += "-detailedSummary:False"

  Invoke-Program $msbuild @MSBuildArgs
}

# TODO(compnerd): replace this with Get-{Build,Host,Target}ProjectBinaryCache
function Get-CMarkBinaryCache($Arch) {
  return "$BinaryCache\$($Arch.LLVMTarget)\cmark-gfm-0.29.0.gfm.13"
}

function Build-CMark($Arch) {
  Build-CMakeProject `
    -Src $SourceCache\cmark `
    -Bin (Get-CMarkBinaryCache $Arch) `
    -InstallTo "$($Arch.ToolchainInstallRoot)\usr" `
    -Arch $Arch `
    -Defines @{
      BUILD_SHARED_LIBS = "YES";
      BUILD_TESTING = "NO";
      CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP = "YES";
    }
}

function Build-BuildTools($Arch) {
  Build-CMakeProject `
    -Src $SourceCache\llvm-project\llvm `
    -Bin (Get-BuildProjectBinaryCache BuildTools) `
    -Arch $Arch `
    -UseMSVCCompilers C,CXX `
    -BuildTargets llvm-tblgen,clang-tblgen,clang-pseudo-gen,clang-tidy-confusable-chars-gen,lldb-tblgen,llvm-config,swift-def-to-strings-converter,swift-serialize-diagnostics,swift-compatibility-symbols `
    -Defines @{
      CMAKE_CROSSCOMPILING = "NO";
      CLANG_ENABLE_LIBXML2 = "NO";
      LLDB_ENABLE_LIBXML2 = "NO";
      LLDB_ENABLE_PYTHON = "NO";
      LLDB_INCLUDE_TESTS = "NO";
      LLDB_ENABLE_SWIFT_SUPPORT = "NO";
      LLVM_ENABLE_ASSERTIONS = "NO";
      LLVM_ENABLE_LIBEDIT = "NO";
      LLVM_ENABLE_LIBXML2 = "NO";
      LLVM_ENABLE_PROJECTS = "clang;clang-tools-extra;lldb";
      LLVM_EXTERNAL_PROJECTS = "swift";
      LLVM_EXTERNAL_SWIFT_SOURCE_DIR = "$SourceCache\swift";
      SWIFT_BUILD_DYNAMIC_SDK_OVERLAY = "NO";
      SWIFT_BUILD_DYNAMIC_STDLIB = "NO";
      SWIFT_BUILD_HOST_DISPATCH = "NO";
      SWIFT_BUILD_LIBEXEC = "NO";
      SWIFT_BUILD_REGEX_PARSER_IN_COMPILER = "NO";
      SWIFT_BUILD_REMOTE_MIRROR = "NO";
      SWIFT_BUILD_SOURCEKIT = "NO";
      SWIFT_BUILD_STATIC_SDK_OVERLAY = "NO";
      SWIFT_BUILD_STATIC_STDLIB = "NO";
      SWIFT_BUILD_SWIFT_SYNTAX = "NO";
      SWIFT_ENABLE_DISPATCH = "NO";
      SWIFT_INCLUDE_APINOTES = "NO";
      SWIFT_INCLUDE_DOCS = "NO";
      SWIFT_INCLUDE_TESTS = "NO";
      "cmark-gfm_DIR" = "$($Arch.ToolchainInstallRoot)\usr\lib\cmake";
    }
}

function Write-PList {
  [CmdletBinding(PositionalBinding = $false)]
  param
  (
    [Parameter(Mandatory = $true)]
    [PSCustomObject] $Settings,
    [Parameter(Mandatory = $true)]
    [string] $Path
  )

  Invoke-Program "$(Get-PythonExecutable)" -c "import plistlib; print(str(plistlib.dumps($(($Settings | ConvertTo-JSON -Compress) -replace '"', "'")), encoding='utf-8'))" `
      -OutFile $Path
}

function Load-LitTestOverrides($Filename) {
  function Select-LitTestOverrides($Prefix) {
    $MatchingLines = Get-Content $Filename | Select-String -Pattern "`^${Prefix}.*$"
    return $MatchingLines | ForEach-Object { ($_ -replace $Prefix,"").Trim() }
  }

  $TestsToXFail = Select-LitTestOverrides "xfail"
  Write-Host "TestsToXFail=$TestsToXFail"
  if ($TestsToXFail -and $TestsToXFail.Length -ne 0) {
    $env:LIT_XFAIL = $TestsToXFail -join ";"
  }
  $TestsToSkip = Select-LitTestOverrides "skip"
  Write-Host "TestsToSkip=$TestsToSkip"
  if ($TestsToSkip -and $TestsToSkip.Length -gt 0) {
    $env:LIT_FILTER_OUT = "($($TestsToSkip -join '|'))"
  }
}

function Build-Compilers() {
  [CmdletBinding(PositionalBinding = $false)]
  param
  (
    [Parameter(Position = 0, Mandatory = $true)]
    [hashtable]$Arch,
    [switch]$TestClang = $false,
    [switch]$TestLLD = $false,
    [switch]$TestLLDB = $false,
    [switch]$TestLLVM = $false,
    [switch]$TestSwift = $false,
    [switch]$Build = $false
  )

  Invoke-IsolatingEnvVars {
    $CompilersBinaryCache = if ($Build) {
      Get-BuildProjectBinaryCache Compilers
    } else {
      Get-HostProjectBinaryCache Compilers
    }
    $BuildTools = Join-Path -Path (Get-BuildProjectBinaryCache BuildTools) -ChildPath bin

    if ($TestClang -or $TestLLD -or $TestLLDB -or $TestLLVM -or $TestSwift) {
      $env:Path = "$(Get-CMarkBinaryCache $Arch)\src;$CompilersBinaryCache\tools\swift\libdispatch-windows-$($Arch.LLVMName)-prefix\bin;$CompilersBinaryCache\bin;$env:Path;$VSInstallRoot\DIA SDK\bin\$($HostArch.VSName);$UnixToolsBinDir"
      $Targets = @()
      $TestingDefines = @{
        SWIFT_BUILD_DYNAMIC_SDK_OVERLAY = "YES";
        SWIFT_BUILD_DYNAMIC_STDLIB = "YES";
        SWIFT_BUILD_REMOTE_MIRROR = "YES";
        SWIFT_NATIVE_SWIFT_TOOLS_PATH = "";
      }

      if ($TestLLVM) { $Targets += @("check-llvm") }
      if ($TestClang) { $Targets += @("check-clang") }
      if ($TestLLD) { $Targets += @("check-lld") }
      if ($TestSwift) { $Targets += @("check-swift", "SwiftCompilerPlugin") }
      if ($TestLLDB) {
        $Targets += @("check-lldb")

        # Override test filter for known issues in downstream LLDB
        Load-LitTestOverrides $PSScriptRoot/windows-llvm-lit-test-overrides.txt

        # Transitive dependency of _lldb.pyd
        $RuntimeBinaryCache = Get-TargetProjectBinaryCache $Arch Runtime
        Copy-Item $RuntimeBinaryCache\bin\swiftCore.dll "$CompilersBinaryCache\lib\site-packages\lldb"

        # Runtime dependencies of repl_swift.exe
        $SwiftRTSubdir = "lib\swift\windows"
        Write-Host "Copying '$RuntimeBinaryCache\$SwiftRTSubdir\$($Arch.LLVMName)\swiftrt.obj' to '$CompilersBinaryCache\$SwiftRTSubdir'"
        Copy-Item "$RuntimeBinaryCache\$SwiftRTSubdir\$($Arch.LLVMName)\swiftrt.obj" "$CompilersBinaryCache\$SwiftRTSubdir"
        Write-Host "Copying '$RuntimeBinaryCache\bin\swiftCore.dll' to '$CompilersBinaryCache\bin'"
        Copy-Item "$RuntimeBinaryCache\bin\swiftCore.dll" "$CompilersBinaryCache\bin"

        $TestingDefines += @{
          LLDB_INCLUDE_TESTS = "YES";
          # Check for required Python modules in CMake
          LLDB_ENFORCE_STRICT_TEST_REQUIREMENTS = "YES";
          # No watchpoint support on windows: https://github.com/llvm/llvm-project/issues/24820
          LLDB_TEST_USER_ARGS = "--skip-category=watchpoint";
          # gtest sharding breaks llvm-lit's --xfail and LIT_XFAIL inputs: https://github.com/llvm/llvm-project/issues/102264
          LLVM_LIT_ARGS = "-v --no-gtest-sharding --show-xfail";
          # LLDB Unit tests link against this library
          LLVM_UNITTEST_LINK_FLAGS = "$(Get-SwiftSDK Windows)\usr\lib\swift\windows\$($Arch.LLVMName)\swiftCore.lib";
        }
      }
    } else {
      $Targets = @("distribution", "install-distribution")
      $TestingDefines = @{
        SWIFT_BUILD_DYNAMIC_SDK_OVERLAY = "NO";
        SWIFT_BUILD_DYNAMIC_STDLIB = "NO";
        SWIFT_BUILD_REMOTE_MIRROR = "NO";
        SWIFT_NATIVE_SWIFT_TOOLS_PATH = $BuildTools;
      }
    }

    $PythonRoot = "$BinaryCache\Python$($Arch.CMakeName)-$PythonVersion\tools"
    $PythonLibName = "python{0}{1}" -f ([System.Version]$PythonVersion).Major, ([System.Version]$PythonVersion).Minor

    # The STL in the latest versions of VS typically require a newer version of
    # Clang than released Swift toolchains include. If bootstrapping with an
    # older toolchain, we need to relax this requirement by defining
    # _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH. Developer builds are (currently)
    # up-to-date.
    $SwiftFlags = @();
    if ([System.Version](Get-PinnedToolchainVersion) -ne [System.Version]"0.0.0") {
      $SwiftFlags += @("-Xcc", "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH");
    }

    # Limit the number of parallel links to avoid OOM when debug info is enabled
    $DebugOptions = @{}
    if ($DebugInfo) {
      $DebugOptions = @{ SWIFT_PARALLEL_LINK_JOBS = "4"; }
    }

    New-Item -ItemType SymbolicLink -Path "$BinaryCache\$($HostArch.LLVMTarget)\compilers" -Target "$BinaryCache\5" -ErrorAction Ignore
    Build-CMakeProject `
      -Src $SourceCache\llvm-project\llvm `
      -Bin $CompilersBinaryCache `
      -InstallTo "$($Arch.ToolchainInstallRoot)\usr" `
      -Arch $Arch `
      -AddAndroidCMakeEnv:$Android `
      -UseMSVCCompilers C,CXX `
      -UsePinnedCompilers Swift `
      -BuildTargets $Targets `
      -CacheScript $SourceCache\swift\cmake\caches\Windows-$($Arch.LLVMName).cmake `
      -Defines ($TestingDefines + @{
        CLANG_TABLEGEN = (Join-Path -Path $BuildTools -ChildPath "clang-tblgen.exe");
        CLANG_TIDY_CONFUSABLE_CHARS_GEN = (Join-Path -Path $BuildTools -ChildPath "clang-tidy-confusable-chars-gen.exe");
        CMAKE_FIND_PACKAGE_PREFER_CONFIG = "YES";
        CMAKE_Swift_FLAGS = $SwiftFlags;
        LibXml2_DIR = "$LibraryRoot\libxml2-2.11.5\usr\lib\Windows\$($Arch.LLVMName)\cmake\libxml2-2.11.5";
        LLDB_PYTHON_EXE_RELATIVE_PATH = "python.exe";
        LLDB_PYTHON_EXT_SUFFIX = ".pyd";
        LLDB_PYTHON_RELATIVE_PATH = "lib/site-packages";
        LLDB_TABLEGEN = (Join-Path -Path $BuildTools -ChildPath "lldb-tblgen.exe");
        LLDB_TEST_MAKE = "$BinaryCache\GnuWin32Make-4.4.1\bin\make.exe";
        LLVM_CONFIG_PATH = (Join-Path -Path $BuildTools -ChildPath "llvm-config.exe");
        LLVM_ENABLE_ASSERTIONS = $(if ($Variant -eq "Asserts") { "YES" } else { "NO" })
        LLVM_EXTERNAL_SWIFT_SOURCE_DIR = "$SourceCache\swift";
        LLVM_HOST_TRIPLE = $BuildArch.LLVMTarget;
        LLVM_NATIVE_TOOL_DIR = $BuildTools;
        LLVM_TABLEGEN = (Join-Path $BuildTools -ChildPath "llvm-tblgen.exe");
        LLVM_USE_HOST_TOOLS = "NO";
        Python3_EXECUTABLE = (Get-PythonExecutable);
        Python3_INCLUDE_DIR = "$PythonRoot\include";
        Python3_LIBRARY = "$PythonRoot\libs\$PythonLibName.lib";
        Python3_ROOT_DIR = $PythonRoot;
        SWIFT_BUILD_SWIFT_SYNTAX = "YES";
        SWIFT_CLANG_LOCATION = (Get-PinnedToolchainToolsDir);
        SWIFT_ENABLE_EXPERIMENTAL_CONCURRENCY = "YES";
        SWIFT_ENABLE_EXPERIMENTAL_CXX_INTEROP = "YES";
        SWIFT_ENABLE_EXPERIMENTAL_DIFFERENTIABLE_PROGRAMMING = "YES";
        SWIFT_ENABLE_EXPERIMENTAL_DISTRIBUTED = "YES";
        SWIFT_ENABLE_EXPERIMENTAL_OBSERVATION = "YES";
        SWIFT_ENABLE_EXPERIMENTAL_STRING_PROCESSING = "YES";
        SWIFT_ENABLE_SYNCHRONIZATION = "YES";
        SWIFT_ENABLE_VOLATILE = "YES";
        SWIFT_PATH_TO_LIBDISPATCH_SOURCE = "$SourceCache\swift-corelibs-libdispatch";
        SWIFT_PATH_TO_STRING_PROCESSING_SOURCE = "$SourceCache\swift-experimental-string-processing";
        SWIFT_PATH_TO_SWIFT_SDK = (Get-PinnedToolchainSDK);
        SWIFT_PATH_TO_SWIFT_SYNTAX_SOURCE = "$SourceCache\swift-syntax";
        SWIFT_STDLIB_ASSERTIONS = "NO";
        SWIFTSYNTAX_ENABLE_ASSERTIONS = "NO";
        "cmark-gfm_DIR" = "$($Arch.ToolchainInstallRoot)\usr\lib\cmake";
      } + $DebugOptions)
  }

  $Settings = @{
    FallbackLibrarySearchPaths = @("usr/bin")
    Identifier = "${ToolchainIdentifier}"
    Version = "${ProductVersion}"
  }

  Write-PList -Settings $Settings -Path "$($Arch.ToolchainInstallRoot)\ToolchainInfo.plist"
}

# Reference: https://github.com/microsoft/mimalloc/tree/dev/bin#minject
function Build-mimalloc() {
  [CmdletBinding(PositionalBinding = $false)]
  param
  (
    [Parameter(Position = 0, Mandatory = $true)]
    [hashtable]$Arch
  )

  # TODO: migrate to the CMake build
  $MSBuildArgs = @()
  $MSBuildArgs += "-noLogo"
  $MSBuildArgs += "-maxCpuCount"

  $Properties = @{}
  Add-KeyValueIfNew $Properties Configuration Release
  Add-KeyValueIfNew $Properties OutDir "$BinaryCache\$($Arch.LLVMTarget)\mimalloc\bin\"
  Add-KeyValueIfNew $Properties Platform "$($Arch.ShortName)"

  Invoke-IsolatingEnvVars {
    Invoke-VsDevShell $Arch
    # Avoid hard-coding the VC tools version number
    $VCRedistDir = (Get-ChildItem "${env:VCToolsRedistDir}\$($HostArch.ShortName)" -Filter "Microsoft.VC*.CRT").FullName
    if ($VCRedistDir) {
      Add-KeyValueIfNew $Properties VCRedistDir "$VCRedistDir\"
    }
  }

  foreach ($Property in $Properties.GetEnumerator()) {
    if ($Property.Value.Contains(" ")) {
      $MSBuildArgs += "-p:$($Property.Key)=$($Property.Value.Replace('\', '\\'))"
    } else {
      $MSBuildArgs += "-p:$($Property.Key)=$($Property.Value)"
    }
  }

  Invoke-Program $msbuild "$SourceCache\mimalloc\ide\vs2022\mimalloc-lib.vcxproj" @MSBuildArgs "-p:IntDir=$BinaryCache\$($Arch.LLVMTarget)\mimalloc\mimalloc\"
  Invoke-Program $msbuild "$SourceCache\mimalloc\ide\vs2022\mimalloc-override-dll.vcxproj" @MSBuildArgs "-p:IntDir=$BinaryCache\$($Arch.LLVMTarget)\mimalloc\mimalloc-override-dll\"

  $HostSuffix = if ($Arch -eq $ArchX64) { "" } else { "-arm64" }
  $BuildSuffix = if ($BuildArch -eq $ArchX64) { "" } else { "-arm64" }

  foreach ($item in "mimalloc.dll", "mimalloc-redirect$HostSuffix.dll") {
    Copy-Item -Path "$BinaryCache\$($Arch.LLVMTarget)\mimalloc\bin\$item" -Destination "$($Arch.ToolchainInstallRoot)\usr\bin\"
  }

  # When cross-compiling, bundle the second mimalloc redirect dll as a workaround for
  # https://github.com/microsoft/mimalloc/issues/997
  if ($IsCrossCompiling) {
    Copy-Item -Path "$BinaryCache\$($Arch.LLVMTarget)\mimalloc\bin\mimalloc-redirect$HostSuffix.dll" -Destination "$($Arch.ToolchainInstallRoot)\usr\bin\mimalloc-redirect$BuildSuffix.dll"
  }

  # TODO: should we split this out into its own function?
  $Tools = @(
    "swift.exe",
    "swiftc.exe",
    "swift-driver.exe",
    "swift-frontend.exe",
    "clang.exe",
    "clang++.exe",
    "clang-cl.exe",
    "lld.exe",
    "lld-link.exe",
    "ld.lld.exe",
    "ld64.lld.exe"
  )
  foreach ($Tool in $Tools) {
    $Binary = [IO.Path]::Combine($Arch.ToolchainInstallRoot, "usr", "bin", $Tool)
    # Binary-patch in place
    Invoke-Program "$SourceCache\mimalloc\bin\minject$BuildSuffix" "-f" "-i" "-v" "$Binary"
    # Log the import table
    $LogFile = "$BinaryCache\$($Arch.LLVMTarget)\mimalloc\minject-log-$Tool.txt"
    $ErrorFile = "$BinaryCache\$($Arch.LLVMTarget)\mimalloc\minject-log-$Tool-error.txt"
    Invoke-Program "$SourceCache\mimalloc\bin\minject$BuildSuffix" "-l" "$Binary" -OutFile $LogFile -ErrorFile $ErrorFile
    # Verify patching
    $Found = Select-String -Path $LogFile -Pattern "mimalloc"
    if (-not $Found) {
      Get-Content $ErrorFile
      throw "Failed to patch mimalloc for $Tool"
    }
  }
}

function Build-LLVM([Platform]$Platform, $Arch) {
  Build-CMakeProject `
    -Src $SourceCache\llvm-project\llvm `
    -Bin (Get-TargetProjectBinaryCache $Arch LLVM) `
    -Arch $Arch `
    -Platform $Platform `
    -UseBuiltCompilers C,CXX `
    -Defines @{
      CMAKE_SYSTEM_NAME = $Platform.ToString();
      LLVM_HOST_TRIPLE = $Arch.LLVMTarget;
    }
}

function Build-Sanitizers([Platform]$Platform, $Arch) {
  $LLVMTargetCache = $(Get-TargetProjectBinaryCache $Arch LLVM)
  $LITVersionStr = $(Invoke-Program $(Get-PythonExecutable) "$LLVMTargetCache\bin\llvm-lit.py" --version)
  if (-not ($LITVersionStr -match "lit (\d+)\.\d+\.\d+.*")) {
    throw "Unexpected version string output from llvm-lit.py"
  }
  $LLVMVersionMajor = $Matches.1
  $InstallTo = "$($HostArch.ToolchainInstallRoot)\usr\lib\clang\$LLVMVersionMajor"
  Write-Host "Sanitizers SDK directory: $InstallTo"

  Build-CMakeProject `
    -Src $SourceCache\llvm-project\compiler-rt\lib\builtins `
    -Bin "$(Get-TargetProjectBinaryCache $Arch ClangBuiltins)" `
    -InstallTo $InstallTo `
    -Arch $Arch `
    -Platform $Platform `
    -UseBuiltCompilers ASM,C,CXX `
    -BuildTargets "install-compiler-rt" `
    -Defines (@{
      CMAKE_SYSTEM_NAME = $Platform.ToString();
      LLVM_DIR = "$LLVMTargetCache\lib\cmake\llvm";
      LLVM_ENABLE_PER_TARGET_RUNTIME_DIR = "YES";
      COMPILER_RT_DEFAULT_TARGET_ONLY = "YES";
    })

  Build-CMakeProject `
    -Src $SourceCache\llvm-project\compiler-rt `
    -Bin "$(Get-TargetProjectBinaryCache $Arch ClangRuntime)" `
    -InstallTo $InstallTo `
    -Arch $Arch `
    -Platform $Platform `
    -UseBuiltCompilers ASM,C,CXX `
    -BuildTargets "install-compiler-rt" `
    -Defines (@{
      CMAKE_SYSTEM_NAME = $Platform.ToString();
      LLVM_DIR = "$LLVMTargetCache\lib\cmake\llvm";
      LLVM_ENABLE_PER_TARGET_RUNTIME_DIR = "YES";
      COMPILER_RT_DEFAULT_TARGET_ONLY = "YES";
      COMPILER_RT_BUILD_BUILTINS = "NO";
      COMPILER_RT_BUILD_CRT = "NO";
      COMPILER_RT_BUILD_LIBFUZZER = "NO";
      COMPILER_RT_BUILD_ORC = "NO";
      COMPILER_RT_BUILD_XRAY = "NO";
      COMPILER_RT_BUILD_PROFILE = "YES";
      COMPILER_RT_BUILD_SANITIZERS = "YES";
    })
}

function Build-ZLib([Platform]$Platform, $Arch) {
  $ArchName = $Arch.LLVMName

  Build-CMakeProject `
    -Src $SourceCache\zlib `
    -Bin "$BinaryCache\$($Arch.LLVMTarget)\zlib-1.3.1" `
    -InstallTo $LibraryRoot\zlib-1.3.1\usr `
    -Arch $Arch `
    -Platform $Platform `
    -UseMSVCCompilers C `
    -Defines @{
      BUILD_SHARED_LIBS = "NO";
      CMAKE_POSITION_INDEPENDENT_CODE = "YES";
      CMAKE_SYSTEM_NAME = $Platform.ToString();
      INSTALL_BIN_DIR = "$LibraryRoot\zlib-1.3.1\usr\bin\$Platform\$ArchName";
      INSTALL_LIB_DIR = "$LibraryRoot\zlib-1.3.1\usr\lib\$Platform\$ArchName";
    }
}

function Build-XML2([Platform]$Platform, $Arch) {
  $ArchName = $Arch.LLVMName

  Build-CMakeProject `
    -Src $SourceCache\libxml2 `
    -Bin "$BinaryCache\$($Arch.LLVMTarget)\libxml2-2.11.5" `
    -InstallTo "$LibraryRoot\libxml2-2.11.5\usr" `
    -Arch $Arch `
    -Platform $Platform `
    -UseMSVCCompilers C,CXX `
    -Defines @{
      BUILD_SHARED_LIBS = "NO";
      CMAKE_INSTALL_BINDIR = "bin/$Platform/$ArchName";
      CMAKE_INSTALL_LIBDIR = "lib/$Platform/$ArchName";
      CMAKE_POSITION_INDEPENDENT_CODE = "YES";
      CMAKE_SYSTEM_NAME = $Platform.ToString();
      LIBXML2_WITH_ICONV = "NO";
      LIBXML2_WITH_ICU = "NO";
      LIBXML2_WITH_LZMA = "NO";
      LIBXML2_WITH_PYTHON = "NO";
      LIBXML2_WITH_TESTS = "NO";
      LIBXML2_WITH_THREADS = "YES";
      LIBXML2_WITH_ZLIB = "NO";
    }
}

function Build-RegsGen2($Arch) {
  Build-CMakeProject `
    -Src $SourceCache\ds2\Tools\RegsGen2 `
    -Bin "$(Get-BuildProjectBinaryCache RegsGen2)" `
    -Arch $Arch `
    -BuildTargets default `
    -UseMSVCCompilers C,CXX `
    -Defines @{
      BISON_EXECUTABLE = "$(Get-BisonExecutable)";
      FLEX_EXECUTABLE = "$(Get-FlexExecutable)";
    }
}

function Build-DS2([Platform]$Platform, $Arch) {
  Build-CMakeProject `
    -Src "$SourceCache\ds2" `
    -Bin "$BinaryCache\$($Arch.LLVMTarget)\ds2" `
    -InstallTo "$(Get-PlatformRoot $Platform)\Developer\Library\ds2\usr" `
    -Arch $Arch `
    -Platform $Platform `
    -Defines @{
      CMAKE_SYSTEM_NAME = $Platform.ToString();
      DS2_REGSGEN2 = "$(Get-BuildProjectBinaryCache RegsGen2)/regsgen2.exe";
      DS2_PROGRAM_PREFIX = "$(Get-ModuleTriple $Arch)-";
      BISON_EXECUTABLE = "$(Get-BisonExecutable)";
      FLEX_EXECUTABLE = "$(Get-FlexExecutable)";
    }
}

function Build-CURL([Platform]$Platform, $Arch) {
  $ArchName = $Arch.LLVMName

  $PlatformDefines = @{}
  if ($Platform -eq "Android") {
    $PlatformDefines += @{
      HAVE_FSEEKO = "0";
    }
  }

  Build-CMakeProject `
    -Src $SourceCache\curl `
    -Bin "$BinaryCache\$($Arch.LLVMTarget)\curl-8.9.1" `
    -InstallTo "$LibraryRoot\curl-8.9.1\usr" `
    -Arch $Arch `
    -Platform $Platform `
    -UseMSVCCompilers C `
    -Defines ($PlatformDefines + @{
      BUILD_SHARED_LIBS = "NO";
      BUILD_TESTING = "NO";
      CMAKE_INSTALL_LIBDIR = "lib/$Platform/$ArchName";
      CMAKE_POSITION_INDEPENDENT_CODE = "YES";
      CMAKE_SYSTEM_NAME = $Platform.ToString();
      BUILD_CURL_EXE = "NO";
      BUILD_LIBCURL_DOCS = "NO";
      BUILD_MISC_DOCS = "NO";
      CURL_CA_BUNDLE = "none";
      CURL_CA_FALLBACK = "NO";
      CURL_CA_PATH = "none";
      CURL_BROTLI = "NO";
      CURL_DISABLE_ALTSVC = "NO";
      CURL_DISABLE_AWS = "YES";
      CURL_DISABLE_BASIC_AUTH = "NO";
      CURL_DISABLE_BEARER_AUTH = "NO";
      CURL_DISABLE_BINDLOCAL = "NO";
      CURL_DISABLE_COOKIES = "NO";
      CURL_DISABLE_DICT = "YES";
      CURL_DISABLE_DIGEST_AUTH = "NO";
      CURL_DISABLE_DOH = "NO";
      CURL_DISABLE_FILE = "YES";
      CURL_DISABLE_FORM_API = "NO";
      CURL_DISABLE_FTP = "YES";
      CURL_DISABLE_GETOPTIONS = "NO";
      CURL_DISABLE_GOPHER = "YES";
      CURL_DISABLE_HEADERS_API = "YES";
      CURL_DISABLE_HSTS = "NO";
      CURL_DISABLE_HTTP = "NO";
      CURL_DISABLE_HTTP_AUTH = "NO";
      CURL_DISABLE_IMAP = "YES";
      CURL_DISABLE_KERBEROS_AUTH = "NO";
      CURL_DISABLE_LDAP = "YES";
      CURL_DISABLE_LDAPS = "YES";
      CURL_DISABLE_MIME = "NO";
      CURL_DISABLE_MQTT = "YES";
      CURL_DISABLE_NEGOTIATE_AUTH = "NO";
      CURL_DISABLE_NETRC = "NO";
      CURL_DISABLE_NTLM = "NO";
      CURL_DISABLE_PARSEDATE = "NO";
      CURL_DISABLE_POP3 = "YES";
      CURL_DISABLE_PROGRESS_METER = "YES";
      CURL_DISABLE_PROXY = "NO";
      CURL_DISABLE_RTSP = "YES";
      CURL_DISABLE_SHUFFLE_DNS = "YES";
      CURL_DISABLE_SMB = "YES";
      CURL_DISABLE_SMTP = "YES";
      CURL_DISABLE_SOCKETPAIR = "YES";
      CURL_DISABLE_SRP = "NO";
      CURL_DISABLE_TELNET = "YES";
      CURL_DISABLE_TFTP = "YES";
      CURL_DISABLE_VERBOSE_STRINGS = "NO";
      CURL_LTO = "NO";
      CURL_USE_BEARSSL = "NO";
      CURL_USE_GNUTLS = "NO";
      CURL_USE_GSSAPI = "NO";
      CURL_USE_LIBPSL = "NO";
      CURL_USE_LIBSSH = "NO";
      CURL_USE_LIBSSH2 = "NO";
      CURL_USE_MBEDTLS = "NO";
      CURL_USE_OPENSSL = "NO";
      CURL_USE_SCHANNEL = if ($Platform -eq "Windows") { "YES" } else { "NO" };
      CURL_USE_WOLFSSL = "NO";
      CURL_WINDOWS_SSPI = if ($Platform -eq "Windows") { "YES" } else { "NO" };
      CURL_ZLIB = "YES";
      CURL_ZSTD = "NO";
      ENABLE_ARES = "NO";
      ENABLE_CURLDEBUG = "NO";
      ENABLE_CURL_MANUAL = "NO";
      ENABLE_DEBUG = "NO";
      ENABLE_IPV6 = "YES";
      ENABLE_THREADED_RESOLVER = "NO";
      ENABLE_UNICODE = "YES";
      ENABLE_UNIX_SOCKETS = "NO";
      ENABLE_WEBSOCKETS = "YES";
      HAVE_POLL_FINE = "NO";
      USE_ECH = "NO";
      USE_HTTPSRR = "NO";
      USE_IDN2 = "NO";
      USE_MSH3 = "NO";
      USE_NGHTTP2 = "NO";
      USE_NGTCP2 = "NO";
      USE_QUICHE = "NO";
      USE_OPENSSL_QUIC = "NO";
      USE_WIN32_IDN = if ($Platform -eq "Windows") { "YES" } else { "NO" };
      USE_WIN32_LARGE_FILES = if ($Platform -eq "Windows") { "YES" } else { "NO" };
      USE_WIN32_LDAP = "NO";
      ZLIB_ROOT = "$LibraryRoot\zlib-1.3.1\usr";
      ZLIB_LIBRARY = "$LibraryRoot\zlib-1.3.1\usr\lib\$Platform\$ArchName\zlibstatic.lib";
    })
}

function Build-Runtime([Platform]$Platform, $Arch) {
  $PlatformDefines = @{}
  if ($Platform -eq [Platform]::Android) {
    $PlatformDefines += @{
      LLVM_ENABLE_LIBCXX = "YES";
      SWIFT_USE_LINKER = "lld";
      # Clang[<18] doesn't provide the _Builtin_float module.
      SWIFT_BUILD_CLANG_OVERLAYS_SKIP_BUILTIN_FLOAT = "YES";
    }
  }

  Invoke-IsolatingEnvVars {
    $env:Path = "$(Get-CMarkBinaryCache $Arch)\src;$(Get-PinnedToolchainRuntime);${env:Path}"

    $CompilersBinaryCache = if ($IsCrossCompiling) {
      Get-BuildProjectBinaryCache Compilers
    } else {
      Get-HostProjectBinaryCache Compilers
    }

    Build-CMakeProject `
      -Src $SourceCache\swift `
      -Bin (Get-TargetProjectBinaryCache $Arch Runtime) `
      -InstallTo "$(Get-SwiftSDK $Platform)\usr" `
      -Arch $Arch `
      -Platform $Platform `
      -CacheScript $SourceCache\swift\cmake\caches\Runtime-$Platform-$($Arch.LLVMName).cmake `
      -UseBuiltCompilers C,CXX,Swift `
      -Defines ($PlatformDefines + @{
        CMAKE_Swift_COMPILER_TARGET = (Get-ModuleTriple $Arch);
        CMAKE_Swift_COMPILER_WORKS = "YES";
        CMAKE_SYSTEM_NAME = $Platform.ToString();
        LLVM_DIR = "$(Get-TargetProjectBinaryCache $Arch LLVM)\lib\cmake\llvm";
        SWIFT_ENABLE_EXPERIMENTAL_CONCURRENCY = "YES";
        SWIFT_ENABLE_EXPERIMENTAL_CXX_INTEROP = "YES";
        SWIFT_ENABLE_EXPERIMENTAL_DIFFERENTIABLE_PROGRAMMING = "YES";
        SWIFT_ENABLE_EXPERIMENTAL_DISTRIBUTED = "YES";
        SWIFT_ENABLE_EXPERIMENTAL_OBSERVATION = "YES";
        SWIFT_ENABLE_EXPERIMENTAL_STRING_PROCESSING = "YES";
        SWIFT_ENABLE_SYNCHRONIZATION = "YES";
        SWIFT_ENABLE_VOLATILE = "YES";
        SWIFT_NATIVE_SWIFT_TOOLS_PATH = (Join-Path -Path $CompilersBinaryCache -ChildPath "bin");
        SWIFT_PATH_TO_LIBDISPATCH_SOURCE = "$SourceCache\swift-corelibs-libdispatch";
        SWIFT_PATH_TO_STRING_PROCESSING_SOURCE = "$SourceCache\swift-experimental-string-processing";
        CMAKE_SHARED_LINKER_FLAGS = if ($Platform -eq "Windows") { @("/INCREMENTAL:NO", "/OPT:REF", "/OPT:ICF") } else { @() };
      })
  }
}

function Test-Runtime([Platform]$Platform, $Arch) {
  if ($IsCrossCompiling) {
    throw "Swift runtime tests are not supported when cross-compiling"
  }
  if (-not (Test-Path (Get-TargetProjectBinaryCache $Arch Runtime))) {
    throw "Swift runtime tests are supposed to reconfigure the existing build"
  }
  $CompilersBinaryCache = Get-HostProjectBinaryCache Compilers
  if (-not (Test-Path "$CompilersBinaryCache\bin\FileCheck.exe")) {
    # These will exist if we test any of llvm/clang/lldb/lld/swift as well
    throw "LIT test utilities not found in $CompilersBinaryCache\bin"
  }

  Invoke-IsolatingEnvVars {
    # Filter known issues when testing on Windows
    Load-LitTestOverrides $PSScriptRoot/windows-swift-android-lit-test-overrides.txt
    $env:Path = "$(Get-CMarkBinaryCache $Arch)\src;$(Get-PinnedToolchainRuntime);${env:Path};$UnixToolsBinDir"
    Build-CMakeProject `
      -Src $SourceCache\swift `
      -Bin (Get-TargetProjectBinaryCache $Arch Runtime) `
      -Arch $Arch `
      -Platform $Platform `
      -UseBuiltCompilers C,CXX,Swift `
      -BuildTargets check-swift-validation-only_non_executable `
      -Defines @{
        SWIFT_INCLUDE_TESTS = "YES";
        SWIFT_INCLUDE_TEST_BINARIES = "YES";
        SWIFT_BUILD_TEST_SUPPORT_MODULES = "YES";
        SWIFT_NATIVE_LLVM_TOOLS_PATH = Join-Path -Path $CompilersBinaryCache -ChildPath "bin";
        LLVM_LIT_ARGS = "-vv";
      }
  }
}

function Build-ExperimentalRuntime {
  [CmdletBinding(PositionalBinding = $false)]
  param
  (
    [Parameter(Position = 0, Mandatory = $true)]
    [Platform] $Platform,
    [Parameter(Position = 1, Mandatory = $true)]
    [hashtable] $Arch,
    [switch] $Static = $false
  )

  # TODO: remove this once the migration is completed.
  Invoke-IsolatingEnvVars {
    Invoke-VsDevShell $BuildArch

    Push-Location "${SourceCache}\swift\Runtimes"
    Start-Process -Wait -WindowStyle Hidden -FilePath cmake.exe -ArgumentList @("-P", "Resync.cmake")
    Pop-Location
  }

  Invoke-IsolatingEnvVars {
    $env:Path = "$(Get-CMarkBinaryCache $Arch)\src;$(Get-PinnedToolchainRuntime);${env:Path}"

    Build-CMakeProject `
      -Src $SourceCache\swift\Runtimes\Core `
      -Bin (Get-TargetProjectBinaryCache $Arch ExperimentalRuntime) `
      -InstallTo "$(Get-SwiftSDK $Platform -Identifier "${Platform}Experimental")\usr" `
      -Arch $Arch `
      -Platform $Platform `
      -UseBuiltCompilers C,CXX,Swift `
      -UseGNUDriver `
      -Defines @{
        BUILD_SHARED_LIBS = if ($Static) { "NO" } else { "YES" };
        CMAKE_FIND_PACKAGE_PREFER_CONFIG = "YES";
        CMAKE_Swift_COMPILER_TARGET = (Get-ModuleTriple $Arch);
        CMAKE_Swift_COMPILER_WORKS = "YES";
        CMAKE_STATIC_LIBRARY_PREFIX_Swift = "lib";
        CMAKE_SYSTEM_NAME = $Platform.ToString();
        dispatch_DIR = "$(Get-TargetProjectBinaryCache $Arch Dispatch)\cmake\modules";
      }
  }
}

function Write-SDKSettingsPlist([Platform]$Platform, $Arch) {
  $SDKSettings = @{
    DefaultProperties = @{
    }
  }
  if ($Platform -eq [Platform]::Windows) {
    $SDKSettings.DefaultProperties.DEFAULT_USE_RUNTIME = "MD"
  }
  Write-PList -Settings $SDKSettings -Path "$(Get-SwiftSDK $Platform)\SDKSettings.plist"

  $SDKSettings = @{
    CanonicalName = "$($Arch.LLVMTarget)"
    DisplayName = "$($Platform.ToString())"
    IsBaseSDK = "NO"
    Version = "${ProductVersion}"
    VersionMap = @{}
    DefaultProperties = @{
      PLATFORM_NAME = "$($Platform.ToString())"
      DEFAULT_COMPILER = "${ToolchainIdentifier}"
    }
  }
  if ($Platform -eq [Platform]::Windows) {
    $SDKSettings.DefaultProperties.DEFAULT_USE_RUNTIME = "MD"
  }
  $SDKSettings | ConvertTo-JSON | Out-FIle -FilePath "$(Get-SwiftSDK $Platform)\SDKSettings.json"
}

function Build-Dispatch([Platform]$Platform, $Arch, [switch]$Test = $false) {
  Invoke-IsolatingEnvVars {
    if ($Test) {
      $Targets = @("default", "ExperimentalTest")
      $InstallPath = ""
      $env:CTEST_OUTPUT_ON_FAILURE = "YES"
    } else {
      $Targets = @("install")
      $InstallPath = "$(Get-SwiftSDK $Platform)\usr"
    }

    Build-CMakeProject `
      -Src $SourceCache\swift-corelibs-libdispatch `
      -Bin (Get-TargetProjectBinaryCache $Arch Dispatch) `
      -InstallTo $InstallPath `
      -Arch $Arch `
      -Platform $Platform `
      -BuildTargets $Targets `
      -UseBuiltCompilers C,CXX,Swift `
      -Defines @{
        ENABLE_SWIFT = "YES";
      }
  }
}

function Build-Foundation {
  [CmdletBinding(PositionalBinding = $false)]
  param
  (
    [Parameter(Position = 0, Mandatory = $true)]
    [Platform] $Platform,
    [Parameter(Position = 1, Mandatory = $true)]
    [hashtable] $Arch,
    [switch] $Static = $false,
    [switch] $Test = $false
  )

  if ($Test) {
    # Foundation tests build via swiftpm rather than CMake
    Build-SPMProject `
      -Action Test `
      -Src $SourceCache\swift-foundation `
      -Bin "$BinaryCache\$($Arch.LLVMTarget)\CoreFoundationTests" `
      -Arch $HostArch

    $ShortArch = $Arch.LLVMName
    Invoke-IsolatingEnvVars {
      $env:DISPATCH_INCLUDE_PATH="$(Get-SwiftSDK $Platform)/usr/include"
      $env:LIBXML_LIBRARY_PATH="$LibraryRoot/libxml2-2.11.5/usr/lib/$Platform/$ShortArch"
      $env:LIBXML_INCLUDE_PATH="$LibraryRoot/libxml2-2.11.5/usr/include/libxml2"
      $env:ZLIB_LIBRARY_PATH="$LibraryRoot/zlib-1.3.1/usr/lib/$Platform/$ShortArch"
      $env:CURL_LIBRARY_PATH="$LibraryRoot/curl-8.9.1/usr/lib/$Platform/$ShortArch"
      $env:CURL_INCLUDE_PATH="$LibraryRoot/curl-8.9.1/usr/include"
      Build-SPMProject `
        -Action Test `
        -Src $SourceCache\swift-corelibs-foundation `
        -Bin "$BinaryCache\$($Arch.LLVMTarget)\FoundationTests" `
        -Arch $HostArch
    }
  } else {
    $DispatchBinaryCache = Get-TargetProjectBinaryCache $Arch Dispatch
    $FoundationBinaryCache = if ($Static) {
      Get-TargetProjectBinaryCache $Arch StaticFoundation
    } else {
      Get-TargetProjectBinaryCache $Arch DynamicFoundation
    }
    $ShortArch = $Arch.LLVMName

    Build-CMakeProject `
      -Src $SourceCache\swift-corelibs-foundation `
      -Bin $FoundationBinaryCache `
      -InstallTo $(if ($Static) { "$(Get-SwiftSDK $Platform -Identifier "${Platform}Experimental")\usr" } else { "$(Get-SwiftSDK $Platform)\usr" }) `
      -Arch $Arch `
      -Platform $Platform `
      -UseBuiltCompilers ASM,C,CXX,Swift `
      -SwiftSDK (Get-SwiftSDK $Platform) `
      -Defines @{
        BUILD_SHARED_LIBS = if ($Static) { "NO" } else { "YES" };
        CMAKE_FIND_PACKAGE_PREFER_CONFIG = "YES";
        CMAKE_NINJA_FORCE_RESPONSE_FILE = "YES";
        CMAKE_STATIC_LIBRARY_PREFIX_Swift = "lib";
        ENABLE_TESTING = "NO";
        FOUNDATION_BUILD_TOOLS = if ($Platform -eq "Windows") { "YES" } else { "NO" };
        CURL_DIR = "$LibraryRoot\curl-8.9.1\usr\lib\$Platform\$ShortArch\cmake\CURL";
        LibXml2_DIR = "$LibraryRoot\libxml2-2.11.5\usr\lib\$Platform\$ShortArch\cmake\libxml2-2.11.5";
        ZLIB_LIBRARY = if ($Platform -eq "Windows") {
          "$LibraryRoot\zlib-1.3.1\usr\lib\$Platform\$ShortArch\zlibstatic.lib"
        } else {
          "$LibraryRoot\zlib-1.3.1\usr\lib\$Platform\$ShortArch\libz.a"
        };
        ZLIB_INCLUDE_DIR = "$LibraryRoot\zlib-1.3.1\usr\include";
        dispatch_DIR = "$DispatchBinaryCache\cmake\modules";
        SwiftSyntax_DIR = (Get-HostProjectCMakeModules Compilers);
        _SwiftFoundation_SourceDIR = "$SourceCache\swift-foundation";
        _SwiftFoundationICU_SourceDIR = "$SourceCache\swift-foundation-icu";
        _SwiftCollections_SourceDIR = "$SourceCache\swift-collections";
        SwiftFoundation_MACRO = "$(Get-BuildProjectBinaryCache FoundationMacros)\bin"
      }
  }
}

function Build-FoundationMacros() {
  [CmdletBinding(PositionalBinding = $false)]
  param
  (
    [Parameter(Position = 0, Mandatory = $true)]
    [Platform]$Platform,
    [Parameter(Position = 1, Mandatory = $true)]
    [hashtable]$Arch,
    [switch] $Build = $false
  )

  $FoundationMacrosBinaryCache = if ($Build) {
    Get-BuildProjectBinaryCache FoundationMacros
  } else {
    Get-HostProjectBinaryCache FoundationMacros
  }

  $SwiftSDK = $null
  if ($Build) {
    $SwiftSDK = $(Get-PinnedToolchainSDK)
  }

  $InstallDir = $null
  $Targets = @("default")
  if (-not $Build) {
    $InstallDir = "$($Arch.ToolchainInstallRoot)\usr"
    $Targets = @()
  }

  $SwiftSyntaxCMakeModules = if ($Build -and $HostArch -ne $BuildArch) {
    Get-BuildProjectCMakeModules Compilers
  } else {
    Get-HostProjectCMakeModules Compilers
  }

  Build-CMakeProject `
    -Src $SourceCache\swift-foundation\Sources\FoundationMacros `
    -Bin $FoundationMacrosBinaryCache `
    -InstallTo:$InstallDir `
    -Arch $Arch `
    -Platform $Platform `
    -UseBuiltCompilers Swift `
    -SwiftSDK:$SwiftSDK `
    -BuildTargets:$Targets `
    -Defines @{
      SwiftSyntax_DIR = $SwiftSyntaxCMakeModules;
    }
}

function Build-XCTest([Platform]$Platform, $Arch) {
  Build-CMakeProject `
    -Src $SourceCache\swift-corelibs-xctest `
    -Bin $(Get-TargetProjectBinaryCache $Arch XCTest) `
    -InstallTo "$([IO.Path]::Combine((Get-PlatformRoot $Platform), "Developer", "Library", "XCTest-development", "usr"))" `
    -Arch $Arch `
    -Platform $Platform `
    -UseBuiltCompilers Swift `
    -Defines @{
      BUILD_SHARED_LIBS = "YES";
      CMAKE_BUILD_WITH_INSTALL_RPATH = "YES";
      CMAKE_INSTALL_BINDIR = $Arch.BinaryDir;
      ENABLE_TESTING = "NO";
      dispatch_DIR = $(Get-TargetProjectCMakeModules $Arch Dispatch);
      Foundation_DIR = $(Get-TargetProjectCMakeModules $Arch DynamicFoundation);
      XCTest_INSTALL_NESTED_SUBDIR = "YES";
    }
}

function Test-XCTest {
  Invoke-IsolatingEnvVars {
    $env:Path = "$(Get-TargetProjectBinaryCache $BuildArch XCTest);$(Get-TargetProjectBinaryCache $BuildArch DynamicFoundation)\bin;$(Get-TargetProjectBinaryCache $BuildArch Dispatch);$(Get-TargetProjectBinaryCache $BuildArch Runtime)\bin;${env:Path};$UnixToolsBinDir"

    Build-CMakeProject `
      -Src $SourceCache\swift-corelibs-xctest `
      -Bin (Get-TargetProjectBinaryCache $BuildArch XCTest) `
      -Arch $BuildArch `
      -Platform Windows `
      -UseBuiltCompilers Swift `
      -BuildTargets default,check-xctest `
      -Defines @{
        CMAKE_BUILD_WITH_INSTALL_RPATH = "YES";
        ENABLE_TESTING = "YES";
        dispatch_DIR = $(Get-TargetProjectCMakeModules $BuildArch Dispatch);
        Foundation_DIR = $(Get-TargetProjectCMakeModules $BuildArch DynamicFoundation);
        LLVM_DIR = "$(Get-TargetProjectBinaryCache $BuildArch LLVM)\lib\cmake\llvm";
        XCTEST_PATH_TO_FOUNDATION_BUILD = $(Get-TargetProjectBinaryCache $BuildArch DynamicFoundation);
        XCTEST_PATH_TO_LIBDISPATCH_BUILD = $(Get-TargetProjectBinaryCache $BuildArch Dispatch);
        XCTEST_PATH_TO_LIBDISPATCH_SOURCE = "$SourceCache\swift-corelibs-libdispatch";
      }
  }
}

function Build-Testing([Platform]$Platform, $Arch) {
  Build-CMakeProject `
    -Src $SourceCache\swift-testing `
    -Bin (Get-TargetProjectBinaryCache $Arch Testing) `
    -InstallTo "$([IO.Path]::Combine((Get-PlatformRoot $Platform), "Developer", "Library", "Testing-development", "usr"))" `
    -Arch $Arch `
    -Platform $Platform `
    -UseBuiltCompilers C,CXX,Swift `
    -Defines @{
      BUILD_SHARED_LIBS = "YES";
      CMAKE_BUILD_WITH_INSTALL_RPATH = "YES";
      CMAKE_INSTALL_BINDIR = $Arch.BinaryDir;
      dispatch_DIR = (Get-TargetProjectCMakeModules $Arch Dispatch);
      Foundation_DIR = (Get-TargetProjectCMakeModules $Arch DynamicFoundation);
      # TODO: ensure that host and target platform match
      SwiftSyntax_DIR = (Get-HostProjectCMakeModules Compilers);
      SwiftTesting_MACRO = "$(Get-BuildProjectBinaryCache TestingMacros)\TestingMacros.dll";
      SwiftTesting_INSTALL_NESTED_SUBDIR = "YES";
    }
}

function Write-PlatformInfoPlist([Platform] $Platform) {
  $Settings = @{
    DefaultProperties = @{
      SWIFT_TESTING_VERSION = "development"
      XCTEST_VERSION = "development"
    }
  }
  if ($Platform -eq [Platform]::Windows) {
    $Settings.DefaultProperties.SWIFTC_FLAGS = @( "-use-ld=lld" )
  }

  Write-PList -Settings $Settings -Path "$(Get-PlatformRoot $Platform)\Info.plist"
}

# Copies files installed by CMake from the arch-specific platform root,
# where they follow the layout expected by the installer,
# to the final platform root, following the installer layout.
function Install-Platform([Platform]$Platform, $Archs) {
  # Copy SDK header files
  foreach ($Module in ("Block", "dispatch", "os", "_foundation_unicode", "_FoundationCShims")) {
    $ModuleDirectory = "$(Get-SwiftSDK $Platform)\usr\lib\swift\$Module"
    if (Test-Path $ModuleDirectory) {
      Move-Directory $ModuleDirectory "$(Get-SwiftSDK $Platform)\usr\include\"
    }
  }

  # Copy files from the arch subdirectory, including "*.swiftmodule" which need restructuring
  foreach ($Arch in $Archs) {
    $PlatformResources = "$(Get-SwiftSDK $Platform)\usr\lib\swift\$($Platform.ToString().ToLowerInvariant())"
    Get-ChildItem -Recurse "$PlatformResources\$($Arch.LLVMName)" | ForEach-Object {
      if (".swiftmodule", ".swiftdoc", ".swiftinterface" -contains $_.Extension) {
        Copy-File $_.FullName "$PlatformResources\$($_.BaseName).swiftmodule\$(Get-ModuleTriple $Arch)$($_.Extension)"
      }
    }
  }
}

function Build-SQLite($Arch) {
  Build-CMakeProject `
    -Src $SourceCache\swift-toolchain-sqlite `
    -Bin "$BinaryCache\$($Arch.LLVMTarget)\sqlite-3.46.0" `
    -InstallTo $LibraryRoot\sqlite-3.46.0\usr `
    -Arch $Arch `
    -UseMSVCCompilers C `
    -Defines @{
      BUILD_SHARED_LIBS = "NO";
    }
}

function Build-System($Arch) {
  Build-CMakeProject `
    -Src $SourceCache\swift-system `
    -Bin (Get-HostProjectBinaryCache System) `
    -Arch $Arch `
    -Platform Windows `
    -UseBuiltCompilers C,Swift `
    -SwiftSDK (Get-SwiftSDK Windows) `
    -BuildTargets default `
    -Defines @{
      BUILD_SHARED_LIBS = "NO";
      CMAKE_STATIC_LIBRARY_PREFIX_Swift = "lib";
    }
}

function Build-Build($Arch) {
  # Use lld to workaround the ARM64 LNK1322 issue: https://github.com/swiftlang/swift/issues/79740
  # FIXME(hjyamauchi) Have a real fix
  $ArchSpecificOptions = if ($Arch -eq $ArchARM64) { @{ CMAKE_Swift_FLAGS = "-use-ld=lld-link"; } } else { @{} }

  Build-CMakeProject `
    -Src $SourceCache\swift-build `
    -Bin (Get-HostProjectBinaryCache Build) `
    -InstallTo "$($Arch.ToolchainInstallRoot)\usr" `
    -Arch $Arch `
    -Platform Windows `
    -UseBuiltCompilers C,CXX,Swift `
    -SwiftSDK (Get-SwiftSDK Windows) `
    -Defines (@{
      BUILD_SHARED_LIBS = "YES";
      CMAKE_STATIC_LIBRARY_PREFIX_Swift = "lib";
      ArgumentParser_DIR = (Get-HostProjectCMakeModules ArgumentParser);
      LLBuild_DIR = (Get-HostProjectCMakeModules LLBuild);
      SwiftDriver_DIR = (Get-HostProjectCMakeModules Driver);
      SwiftSystem_DIR = (Get-HostProjectCMakeModules System);
      TSC_DIR = (Get-HostProjectCMakeModules ToolsSupportCore);
      SQLite3_INCLUDE_DIR = "$LibraryRoot\sqlite-3.46.0\usr\include";
      SQLite3_LIBRARY = "$LibraryRoot\sqlite-3.46.0\usr\lib\SQLite3.lib";
    } + $ArchSpecificOptions)
}

function Build-ToolsSupportCore($Arch) {
  Build-CMakeProject `
    -Src $SourceCache\swift-tools-support-core `
    -Bin (Get-HostProjectBinaryCache ToolsSupportCore) `
    -InstallTo "$($Arch.ToolchainInstallRoot)\usr" `
    -Arch $Arch `
    -Platform Windows `
    -UseBuiltCompilers C,Swift `
    -SwiftSDK (Get-SwiftSDK Windows) `
    -Defines @{
      BUILD_SHARED_LIBS = "YES";
      CMAKE_STATIC_LIBRARY_PREFIX_Swift = "lib";
    }
}

function Build-LLBuild($Arch, [switch]$Test = $false) {
  Invoke-IsolatingEnvVars {
    if ($Test) {
      # Build additional llvm executables needed by tests
      Invoke-IsolatingEnvVars {
        Invoke-VsDevShell $HostArch
        Invoke-Program ninja.exe -C (Get-BuildProjectBinaryCache BuildTools) FileCheck not
      }

      $Targets = @("default", "test-llbuild")
      $TestingDefines = @{
        FILECHECK_EXECUTABLE = ([IO.Path]::Combine((Get-BuildProjectBinaryCache BuildTools), "bin", "FileCheck.exe"));
        LIT_EXECUTABLE = "$SourceCache\llvm-project\llvm\utils\lit\lit.py";
      }
      $env:Path = "$env:Path;$UnixToolsBinDir"
      $env:AR = ([IO.Path]::Combine((Get-HostProjectBinaryCache Compilers), "bin", "llvm-ar.exe"))
      $env:CLANG = ([IO.Path]::Combine((Get-HostProjectBinaryCache Compilers), "bin", "clang.exe"))
      $InstallPath = ""
    } else {
      $Targets = @()
      $TestingDefines = @{}
      $InstallPath = "$($Arch.ToolchainInstallRoot)\usr"
    }

    Build-CMakeProject `
      -Src $SourceCache\llbuild `
      -Bin (Get-HostProjectBinaryCache LLBuild) `
      -InstallTo $InstallPath `
      -Arch $Arch `
      -Platform Windows `
      -UseMSVCCompilers CXX `
      -UseBuiltCompilers Swift `
      -SwiftSDK (Get-SwiftSDK Windows) `
      -BuildTargets $Targets `
      -Defines ($TestingDefines + @{
        BUILD_SHARED_LIBS = "YES";
        LLBUILD_SUPPORT_BINDINGS = "Swift";
        SQLite3_INCLUDE_DIR = "$LibraryRoot\sqlite-3.46.0\usr\include";
        SQLite3_LIBRARY = "$LibraryRoot\sqlite-3.46.0\usr\lib\SQLite3.lib";
      })
  }
}

function Build-ArgumentParser($Arch) {
  Build-CMakeProject `
    -Src $SourceCache\swift-argument-parser `
    -Bin (Get-HostProjectBinaryCache ArgumentParser) `
    -InstallTo "$($Arch.ToolchainInstallRoot)\usr" `
    -Arch $Arch `
    -Platform Windows `
    -UseBuiltCompilers Swift `
    -SwiftSDK (Get-SwiftSDK Windows) `
    -Defines @{
      BUILD_SHARED_LIBS = "YES";
      BUILD_TESTING = "NO";
      CMAKE_STATIC_LIBRARY_PREFIX_Swift = "lib";
    }
}

function Build-Driver($Arch) {
  Build-CMakeProject `
    -Src $SourceCache\swift-driver `
    -Bin (Get-HostProjectBinaryCache Driver) `
    -InstallTo "$($Arch.ToolchainInstallRoot)\usr" `
    -Arch $Arch `
    -Platform Windows `
    -UseBuiltCompilers C,CXX,Swift `
    -SwiftSDK (Get-SwiftSDK Windows) `
    -Defines @{
      BUILD_SHARED_LIBS = "YES";
      CMAKE_STATIC_LIBRARY_PREFIX_Swift = "lib";
      TSC_DIR = (Get-HostProjectCMakeModules ToolsSupportCore);
      LLBuild_DIR = (Get-HostProjectCMakeModules LLBuild);
      ArgumentParser_DIR = (Get-HostProjectCMakeModules ArgumentParser);
      SQLite3_INCLUDE_DIR = "$LibraryRoot\sqlite-3.46.0\usr\include";
      SQLite3_LIBRARY = "$LibraryRoot\sqlite-3.46.0\usr\lib\SQLite3.lib";
      SWIFT_DRIVER_BUILD_TOOLS = "YES";
      LLVM_DIR = "$(Get-HostProjectBinaryCache Compilers)\lib\cmake\llvm";
      Clang_DIR = "$(Get-HostProjectBinaryCache Compilers)\lib\cmake\clang";
      Swift_DIR = "$(Get-HostProjectBinaryCache Compilers)\tools\swift\lib\cmake\swift";
    }
}

function Build-Crypto($Arch) {
  Build-CMakeProject `
    -Src $SourceCache\swift-crypto `
    -Bin (Get-HostProjectBinaryCache Crypto) `
    -Arch $Arch `
    -Platform Windows `
    -UseBuiltCompilers Swift `
    -SwiftSDK (Get-SwiftSDK Windows) `
    -BuildTargets default `
    -Defines @{
      BUILD_SHARED_LIBS = "NO";
      CMAKE_STATIC_LIBRARY_PREFIX_Swift = "lib";
    }
}

function Build-Collections($Arch) {
  Build-CMakeProject `
    -Src $SourceCache\swift-collections `
    -Bin (Get-HostProjectBinaryCache Collections) `
    -InstallTo "$($Arch.ToolchainInstallRoot)\usr" `
    -Arch $Arch `
    -Platform Windows `
    -UseBuiltCompilers C,Swift `
    -SwiftSDK (Get-SwiftSDK Windows) `
    -Defines @{
      BUILD_SHARED_LIBS = "YES";
      CMAKE_STATIC_LIBRARY_PREFIX_Swift = "lib";
    }
}

function Build-ASN1($Arch) {
  Build-CMakeProject `
    -Src $SourceCache\swift-asn1 `
    -Bin (Get-HostProjectBinaryCache ASN1) `
    -Arch $Arch `
    -UseBuiltCompilers Swift `
    -SwiftSDK (Get-SwiftSDK Windows) `
    -BuildTargets default `
    -Defines @{
      BUILD_SHARED_LIBS = "NO";
      CMAKE_STATIC_LIBRARY_PREFIX_Swift = "lib";
    }
}

function Build-Certificates($Arch) {
  Build-CMakeProject `
    -Src $SourceCache\swift-certificates `
    -Bin (Get-HostProjectBinaryCache Certificates) `
    -Arch $Arch `
    -Platform Windows `
    -UseBuiltCompilers Swift `
    -SwiftSDK (Get-SwiftSDK Windows) `
    -BuildTargets default `
    -Defines @{
      BUILD_SHARED_LIBS = "NO";
      CMAKE_STATIC_LIBRARY_PREFIX_Swift = "lib";
      SwiftCrypto_DIR = (Get-HostProjectCMakeModules Crypto);
      SwiftASN1_DIR = (Get-HostProjectCMakeModules ASN1);
    }
}

function Build-PackageManager($Arch) {
  $SrcDir = if (Test-Path -Path "$SourceCache\swift-package-manager" -PathType Container) {
    "$SourceCache\swift-package-manager"
  } else {
    "$SourceCache\swiftpm"
  }

  Build-CMakeProject `
    -Src $SrcDir `
    -Bin (Get-HostProjectBinaryCache PackageManager) `
    -InstallTo "$($Arch.ToolchainInstallRoot)\usr" `
    -Arch $Arch `
    -Platform Windows `
    -UseBuiltCompilers C,Swift `
    -SwiftSDK (Get-SwiftSDK Windows) `
    -Defines @{
      BUILD_SHARED_LIBS = "YES";
      CMAKE_Swift_FLAGS = @("-DCRYPTO_v2");
      CMAKE_STATIC_LIBRARY_PREFIX_Swift = "lib";
      SwiftSystem_DIR = (Get-HostProjectCMakeModules System);
      TSC_DIR = (Get-HostProjectCMakeModules ToolsSupportCore);
      LLBuild_DIR = (Get-HostProjectCMakeModules LLBuild);
      ArgumentParser_DIR = (Get-HostProjectCMakeModules ArgumentParser);
      SwiftDriver_DIR = (Get-HostProjectCMakeModules Driver);
      SwiftCrypto_DIR = (Get-HostProjectCMakeModules Crypto);
      SwiftCollections_DIR = (Get-HostProjectCMakeModules Collections);
      SwiftASN1_DIR = (Get-HostProjectCMakeModules ASN1);
      SwiftCertificates_DIR = (Get-HostProjectCMakeModules Certificates);
      SwiftSyntax_DIR = (Get-HostProjectCMakeModules Compilers);
      SQLite3_INCLUDE_DIR = "$LibraryRoot\sqlite-3.46.0\usr\include";
      SQLite3_LIBRARY = "$LibraryRoot\sqlite-3.46.0\usr\lib\SQLite3.lib";
    }
}

function Build-Markdown($Arch) {
  Build-CMakeProject `
    -Src $SourceCache\swift-markdown `
    -Bin (Get-HostProjectBinaryCache Markdown) `
    -InstallTo "$($Arch.ToolchainInstallRoot)\usr" `
    -Arch $Arch `
    -Platform Windows `
    -UseBuiltCompilers C,Swift `
    -SwiftSDK (Get-SwiftSDK Windows) `
    -Defines @{
      BUILD_SHARED_LIBS = "NO";
      CMAKE_STATIC_LIBRARY_PREFIX_Swift = "lib";
      ArgumentParser_DIR = (Get-HostProjectCMakeModules ArgumentParser);
      "cmark-gfm_DIR" = "$($Arch.ToolchainInstallRoot)\usr\lib\cmake";
    }
}

function Build-Format($Arch) {
  Build-CMakeProject `
    -Src $SourceCache\swift-format `
    -Bin (Get-HostProjectBinaryCache Format) `
    -InstallTo "$($Arch.ToolchainInstallRoot)\usr" `
    -Arch $Arch `
    -Platform Windows `
    -UseMSVCCompilers C `
    -UseBuiltCompilers Swift `
    -SwiftSDK (Get-SwiftSDK Windows) `
    -Defines @{
      BUILD_SHARED_LIBS = "YES";
      ArgumentParser_DIR = (Get-HostProjectCMakeModules ArgumentParser);
      SwiftSyntax_DIR = (Get-HostProjectCMakeModules Compilers);
      SwiftMarkdown_DIR = (Get-HostProjectCMakeModules Markdown);
      "cmark-gfm_DIR" = "$($Arch.ToolchainInstallRoot)\usr\lib\cmake";
    }
}

function Test-Format {
  $SwiftPMArguments = @(
    # swift-syntax
    "-Xswiftc", "-I$(Get-HostProjectBinaryCache Compilers)\lib\swift\host",
    "-Xswiftc", "-L$(Get-HostProjectBinaryCache Compilers)\lib\swift\host",
    # swift-argument-parser
    "-Xswiftc", "-I$(Get-HostProjectBinaryCache ArgumentParser)\swift",
    "-Xlinker", "-L$(Get-HostProjectBinaryCache ArgumentParser)\lib",
    # swift-cmark
    "-Xswiftc", "-I$($SourceCache)\cmark\src\include",
    "-Xswiftc", "-I$($SourceCache)\cmark\extensions\include",
    "-Xlinker", "-I$($SourceCache)\cmark\extensions\include",
    "-Xlinker", "$(Get-CMarkBinaryCache $HostArch)\src\cmark-gfm.lib",
    "-Xlinker", "$(Get-CMarkBinaryCache $HostArch)\extensions\cmark-gfm-extensions.lib",
    # swift-markdown
    "-Xlinker", "$(Get-HostProjectBinaryCache Markdown)\lib\CAtomic.lib",
    "-Xswiftc", "-I$($SourceCache)\swift-markdown\Sources\CAtomic\include",
    "-Xswiftc", "-I$(Get-HostProjectBinaryCache Markdown)\swift",
    "-Xlinker", "-L$(Get-HostProjectBinaryCache Markdown)\lib",
    # swift-format
    "-Xswiftc", "-I$(Get-HostProjectBinaryCache Format)\swift",
    "-Xlinker", "-L$(Get-HostProjectBinaryCache Format)\lib"
  )

  Invoke-IsolatingEnvVars {
    $env:SWIFTFORMAT_BUILD_ONLY_TESTS=1
    # Testing swift-format is faster in serial mode than in parallel mode, probably because parallel test execution
    # launches a process for every test class and the process launching overhead on Windows is greater than any
    # gains from parallel test execution.
    Build-SPMProject `
      -Action Test `
      -Src "$SourceCache\swift-format" `
      -Bin "$BinaryCache\$($HostArch.LLVMTarget)\FormatTests" `
      -Arch $HostArch `
      @SwiftPMArguments
  }
}

function Build-LMDB($Arch) {
  Build-CMakeProject `
    -Src $SourceCache\swift-lmdb `
    -Bin (Get-HostProjectBinaryCache LMDB) `
    -Arch $Arch `
    -UseMSVCCompilers C `
    -BuildTargets default
}

function Build-IndexStoreDB($Arch) {
  Build-CMakeProject `
    -Src $SourceCache\indexstore-db `
    -Bin (Get-HostProjectBinaryCache IndexStoreDB) `
    -Arch $Arch `
    -Platform Windows `
    -UseBuiltCompilers C,CXX,Swift `
    -SwiftSDK (Get-SwiftSDK Windows) `
    -BuildTargets default `
    -Defines @{
      BUILD_SHARED_LIBS = "NO";
      CMAKE_STATIC_LIBRARY_PREFIX_Swift = "lib";
      CMAKE_C_FLAGS = @("-I$(Get-SwiftSDK Windows)\usr\include", "-I$(Get-SwiftSDK Windows)\usr\include\Block");
      CMAKE_CXX_FLAGS = @("-I$(Get-SwiftSDK Windows)\usr\include", "-I$(Get-SwiftSDK Windows)\usr\include\Block");
      LMDB_DIR = (Get-HostProjectCMakeModules LMDB);
    }
}

function Build-SourceKitLSP($Arch) {
  Build-CMakeProject `
    -Src $SourceCache\sourcekit-lsp `
    -Bin (Get-HostProjectBinaryCache SourceKitLSP) `
    -InstallTo "$($Arch.ToolchainInstallRoot)\usr" `
    -Arch $Arch `
    -Platform Windows `
    -UseBuiltCompilers C,Swift `
    -SwiftSDK (Get-SwiftSDK Windows) `
    -Defines @{
      CMAKE_STATIC_LIBRARY_PREFIX_Swift = "lib";
      SwiftSyntax_DIR = (Get-HostProjectCMakeModules Compilers);
      TSC_DIR = (Get-HostProjectCMakeModules ToolsSupportCore);
      LLBuild_DIR = (Get-HostProjectCMakeModules LLBuild);
      ArgumentParser_DIR = (Get-HostProjectCMakeModules ArgumentParser);
      SwiftCrypto_DIR = (Get-HostProjectCMakeModules Crypto);
      SwiftCollections_DIR = (Get-HostProjectCMakeModules Collections);
      SwiftPM_DIR = (Get-HostProjectCMakeModules PackageManager);
      LMDB_DIR = (Get-HostProjectCMakeModules LMDB);
      IndexStoreDB_DIR = (Get-HostProjectCMakeModules IndexStoreDB);
    }
}

function Test-SourceKitLSP {
  $SwiftPMArguments = @(
    # dispatch
    "-Xcc", "-I$SourceCache\swift-corelibs-libdispatch",
    "-Xcc", "-I$SourceCache\swift-corelibs-libdispatch\src\BlocksRuntime",
    # swift-syntax
    "-Xswiftc", "-I$(Get-HostProjectBinaryCache Compilers)\lib\swift\host",
    "-Xswiftc", "-L$(Get-HostProjectBinaryCache Compilers)\lib\swift\host",
    # swift-cmark
    "-Xswiftc", "-I$SourceCache\cmark\src\include",
    "-Xswiftc", "-I$SourceCache\cmark\extensions\include",
    "-Xlinker", "-I$SourceCache\cmark\extensions\include",
    "-Xlinker", "$(Get-CMarkBinaryCache $HostArch)\src\cmark-gfm.lib",
    "-Xlinker", "$(Get-CMarkBinaryCache $HostArch)\extensions\cmark-gfm-extensions.lib",
    # swift-system
    "-Xswiftc", "-I$($SourceCache)\swift-system\Sources\CSystem\include",
    "-Xswiftc", "-I$(Get-HostProjectBinaryCache System)\swift",
    "-Xlinker", "-L$(Get-HostProjectBinaryCache System)\lib",
    # swift-tools-support-core
    "-Xswiftc", "-I$(Get-HostProjectBinaryCache ToolsSupportCore)\swift",
    "-Xlinker", "-L$(Get-HostProjectBinaryCache ToolsSupportCore)\lib",
    # swift-llbuild
    "-Xswiftc", "-I$($SourceCache)\llbuild\products\libllbuild\include",
    "-Xswiftc", "-I$(Get-HostProjectBinaryCache LLBuild)\products\llbuildSwift",
    "-Xlinker", "-L$(Get-HostProjectBinaryCache LLBuild)\lib",
    # swift-argument-parser
    "-Xswiftc", "-I$(Get-HostProjectBinaryCache ArgumentParser)\swift",
    "-Xlinker", "-L$(Get-HostProjectBinaryCache ArgumentParser)\lib",
    # swift-crypto
    "-Xswiftc", "-I$(Get-HostProjectBinaryCache Crypto)\swift",
    "-Xlinker", "-L$(Get-HostProjectBinaryCache Crypto)\lib",
    "-Xlinker", "$(Get-HostProjectBinaryCache Crypto)\lib\CCryptoBoringSSL.lib",
    # swift-package-manager
    "-Xswiftc", "-I$(Get-HostProjectBinaryCache PackageManager)\swift",
    "-Xlinker", "-L$(Get-HostProjectBinaryCache PackageManager)\lib",
    # swift-markdown
    "-Xswiftc", "-I$($SourceCache)\swift-markdown\Sources\CAtomic\inclde",
    "-Xlinker", "$(Get-HostProjectBinaryCache Markdown)\lib\CAtomic.lib",
    "-Xswiftc", "-I$(Get-HostProjectBinaryCache Markdown)\swift",
    "-Xlinker", "-L$(Get-HostProjectBinaryCache Markdown)\lib",
    # swift-format
    "-Xswiftc", "-I$(Get-HostProjectBinaryCache Format)\swift",
    "-Xlinker", "-L$(Get-HostProjectBinaryCache Format)\lib",
    # indexstore-db
    "-Xswiftc", "-I$(Get-HostProjectBinaryCache IndexStoreDB)\swift",
    "-Xlinker", "-L$(Get-HostProjectBinaryCache IndexStoreDB)\Sources\IndexStoreDB",
    "-Xlinker", "$(Get-HostProjectBinaryCache IndexStoreDB)\Sources\IndexStoreDB_CIndexStoreDB\CIndexStoreDB.lib",
    "-Xlinker", "$(Get-HostProjectBinaryCache IndexStoreDB)\Sources\IndexStoreDB_Core\Core.lib",
    "-Xlinker", "$(Get-HostProjectBinaryCache IndexStoreDB)\Sources\IndexStoreDB_Database\Database.lib",
    "-Xlinker", "$(Get-HostProjectBinaryCache IndexStoreDB)\Sources\IndexStoreDB_Index\Index.lib",
    "-Xlinker", "$(Get-HostProjectBinaryCache IndexStoreDB)\Sources\IndexStoreDB_LLVMSupport\LLVMSupport.lib",
    "-Xlinker", "$(Get-HostProjectBinaryCache IndexStoreDB)\Sources\IndexStoreDB_Support\Support.lib",
    # LMDB
    "-Xlinker", "$(Get-HostProjectBinaryCache LMDB)\lib\CLMDB.lib",
    # sourcekit-lsp
    "-Xswiftc", "-I$($SourceCache)\sourcekit-lsp\Sources\CAtomics\include",
    "-Xswiftc", "-I$($SourceCache)\sourcekit-lsp\Sources\CSourcekitd\include",
    "-Xlinker", "$(Get-HostProjectBinaryCache SourceKitLSP)\lib\CSourcekitd.lib",
    "-Xswiftc", "-I$($SourceCache)\sourcekit-lsp\Sources\CCompletionScoring\include",
    "-Xswiftc", "-I$(Get-HostProjectBinaryCache SourceKitLSP)\swift",
    "-Xlinker", "-L$(Get-HostProjectBinaryCache SourceKitLSP)\lib"
  )

  Invoke-IsolatingEnvVars {
    $env:SOURCEKIT_LSP_BUILD_ONLY_TESTS=1

    # CI doesn't contain any sensitive information. Log everything.
    $env:SOURCEKIT_LSP_LOG_PRIVACY_LEVEL="sensitive"

    # Log with the highest log level to simplify debugging of CI failures.
    $env:SOURCEKIT_LSP_LOG_LEVEL="debug"

    # The Windows build doesn't build the SourceKit plugins into the SwiftPM build directory (it builds them using CMake).
    # Tell the tests where to find the just-built plugins.
    $env:SOURCEKIT_LSP_TEST_PLUGIN_PATHS="$($HostArch.ToolchainInstallRoot)\usr\lib"

    Build-SPMProject `
      -Action TestParallel `
      -Src "$SourceCache\sourcekit-lsp" `
      -Bin "$BinaryCache\$($HostArch.LLVMTarget)\SourceKitLSPTests" `
      -Arch $HostArch `
      @SwiftPMArguments
  }
}

function Build-TestingMacros() {
  [CmdletBinding(PositionalBinding = $false)]
  param
  (
    [Parameter(Position = 0, Mandatory = $true)]
    [Platform]$Platform,
    [Parameter(Position = 1, Mandatory = $true)]
    [hashtable]$Arch,
    [switch] $Build = $false
  )

  $TestingMacrosBinaryCache = if ($Build) {
    Get-BuildProjectBinaryCache TestingMacros
  } else {
    Get-HostProjectBinaryCache TestingMacros
  }

  $SwiftSDK = $null
  if ($Build) {
    $SwiftSDK = $(Get-PinnedToolchainSDK)
  }

  $Targets = if ($Build) {
    @("default")
  } else {
    @("default", "install")
  }

  $InstallDir = $null
  $Targets = @("default")
  if (-not $Build) {
    $InstallDir = "$($Arch.ToolchainInstallRoot)\usr"
    $Targets = @()
  }

  $SwiftSyntaxCMakeModules = if ($Build -and $HostArch -ne $BuildArch) {
    Get-BuildProjectCMakeModules Compilers
  } else {
    Get-HostProjectCMakeModules Compilers
  }

  Build-CMakeProject `
    -Src $SourceCache\swift-testing\Sources\TestingMacros `
    -Bin $TestingMacrosBinaryCache `
    -InstallTo:$InstallDir  `
    -Arch $Arch `
    -Platform $Platform `
    -UseBuiltCompilers Swift `
    -SwiftSDK:$SwiftSDK `
    -BuildTargets:$Targets `
    -Defines @{
      SwiftSyntax_DIR = $SwiftSyntaxCMakeModules;
    }
}

function Install-HostToolchain() {
  if ($ToBatch) { return }

  # We've already special-cased $HostArch.ToolchainInstallRoot to point to $ToolchainInstallRoot.
  # There are only a few extra restructuring steps we need to take care of.

  # Restructure _InternalSwiftScan (keep the original one for the installer)
  Copy-Item -Force `
    "$($HostArch.ToolchainInstallRoot)\usr\lib\swift\_InternalSwiftScan" `
    "$($HostArch.ToolchainInstallRoot)\usr\include"
  Copy-Item -Force `
    "$($HostArch.ToolchainInstallRoot)\usr\lib\swift\windows\_InternalSwiftScan.lib" `
    "$($HostArch.ToolchainInstallRoot)\usr\lib"

  # Switch to swift-driver
  $SwiftDriver = ([IO.Path]::Combine((Get-HostProjectBinaryCache Driver), "bin", "swift-driver.exe"))
  Copy-Item -Force $SwiftDriver "$($HostArch.ToolchainInstallRoot)\usr\bin\swift.exe"
  Copy-Item -Force $SwiftDriver "$($HostArch.ToolchainInstallRoot)\usr\bin\swiftc.exe"
}

function Build-Inspect([Platform]$Platform, $Arch) {
  if ($Arch -eq $HostArch) {
    # When building for the host target, use the host version of the swift-argument-parser,
    # and place the host swift-inspect executable with the other host toolchain binaries.
    $ArgumentParserDir = Get-HostProjectCMakeModules ArgumentParser
    $InstallPath = "$($HostArch.ToolchainInstallRoot)\usr"
  } else {
    # When building for non-host target, let CMake fetch the swift-argument-parser dependency
    # since it is currently only built for the host and and cannot be built for Android until
    # the pinned version is >= 1.5.0.
    $ArgumentParserDir = ""
    $InstallPath = "$(Get-PlatformRoot $Platform)\Developer\Library\$(Get-ModuleTriple $Arch)"
  }

  Build-CMakeProject `
    -Src $SourceCache\swift\tools\swift-inspect `
    -Bin (Get-TargetProjectBinaryCache $Arch SwiftInspect)`
    -InstallTo $InstallPath `
    -Arch $Arch `
    -Platform $Platform `
    -UseBuiltCompilers C,CXX,Swift `
    -SwiftSDK (Get-SwiftSDK $Platform) `
    -Defines @{
      CMAKE_Swift_FLAGS = @(
        "-Xcc", "-I$(Get-SwiftSDK $Platform)\usr\lib\swift",
        "-Xcc", "-I$(Get-SwiftSDK $Platform)\usr\include\swift\SwiftRemoteMirror",
        "-L$(Get-SwiftSDK $Platform)\usr\lib\swift\$Platform");
      ArgumentParser_DIR = $ArgumentParserDir;
    }
}

function Build-DocC() {
  Build-SPMProject `
    -Action Build `
    -Src $SourceCache\swift-docc `
    -Bin $(Get-HostProjectBinaryCache DocC) `
    -Arch $HostArch `
    --product docc
}

function Test-PackageManager() {
  $SrcDir = if (Test-Path -Path "$SourceCache\swift-package-manager" -PathType Container) {
    "$SourceCache\swift-package-manager"
  } else {
    "$SourceCache\swiftpm"
  }

  Build-SPMProject `
    -Action Test `
    -Src $SrcDir `
    -Bin "$BinaryCache\$($HostArch.LLVMTarget)\PackageManagerTests" `
    -Arch $HostArch `
    -Xcc "-I$LibraryRoot\sqlite-3.46.0\usr\include" -Xlinker "-L$LibraryRoot\sqlite-3.46.0\usr\lib"
}

function Build-Installer($Arch) {
  # TODO(hjyamauchi) Re-enable the swift-inspect and swift-docc builds
  # when cross-compiling https://github.com/apple/swift/issues/71655
  $INCLUDE_SWIFT_DOCC = if ($IsCrossCompiling) { "false" } else { "true" }

  $Properties = @{
    BundleFlavor = "offline";
    TOOLCHAIN_ROOT = "$($Arch.ToolchainInstallRoot)\";
    # When cross-compiling, bundle the second mimalloc redirect dll as a workaround for
    # https://github.com/microsoft/mimalloc/issues/997
    WORKAROUND_MIMALLOC_ISSUE_997 = if ($IsCrossCompiling) { "true" } else { "false" };
    INCLUDE_SWIFT_DOCC = $INCLUDE_SWIFT_DOCC;
    SWIFT_DOCC_BUILD = "$(Get-HostProjectBinaryCache DocC)\release";
    SWIFT_DOCC_RENDER_ARTIFACT_ROOT = "${SourceCache}\swift-docc-render-artifact";
  }

  Invoke-IsolatingEnvVars {
    Invoke-VsDevShell $Arch
    # Avoid hard-coding the VC tools version number
    $VCRedistDir = (Get-ChildItem "${env:VCToolsRedistDir}\$($HostArch.ShortName)" -Filter "Microsoft.VC*.CRT").FullName
    if ($VCRedistDir) {
      $Properties["VCRedistDir"] = "$VCRedistDir\"
    }
  }

  foreach ($SDK in $WindowsSDKArchs) {
    $Properties["INCLUDE_WINDOWS_$($SDK.VSName.ToUpperInvariant())_SDK"] = "true"
    $Properties["PLATFORM_ROOT_$($SDK.VSName.ToUpperInvariant())"] = "$(Get-PlatformRoot Windows)\";
    $Properties["SDK_ROOT_$($SDK.VSName.ToUpperInvariant())"] = "$(Get-SwiftSDK Windows)\"
  }

  Build-WiXProject bundle\installer.wixproj -Arch $Arch -Bundle -Properties $Properties
}

function Copy-BuildArtifactsToStage($Arch) {
  Copy-File "$BinaryCache\$($Arch.LLVMTarget)\installer\Release\$($Arch.VSName)\*.cab" $Stage
  Copy-File "$BinaryCache\$($Arch.LLVMTarget)\installer\Release\$($Arch.VSName)\*.msi" $Stage
  foreach ($SDK in $WindowsSDKArchs) {
    Copy-File "$BinaryCache\$($Arch.LLVMTarget)\installer\Release\$($SDK.VSName)\sdk.windows.$($SDK.VSName).cab" $Stage
    Copy-File "$BinaryCache\$($Arch.LLVMTarget)\installer\Release\$($SDK.VSName)\sdk.windows.$($SDK.VSName).msi" $Stage
    Copy-File "$BinaryCache\$($Arch.LLVMTarget)\installer\Release\$($SDK.VSName)\rtl.$($SDK.VSName).msm" $Stage
  }
  Copy-File "$BinaryCache\$($Arch.LLVMTarget)\installer\Release\$($Arch.VSName)\installer.exe" $Stage
  # Extract installer engine to ease code-signing on swift.org CI
  if ($ToBatch) {
    Write-Output "md `"$BinaryCache\$($Arch.LLVMTarget)\installer\$($Arch.VSName)`""
  } else {
    New-Item -Type Directory -Path "$BinaryCache\$($Arch.LLVMTarget)\installer\$($Arch.VSName)" -ErrorAction Ignore | Out-Null
  }
  Invoke-Program "$($WiX.Path)\wix.exe" -- burn detach "$BinaryCache\$($Arch.LLVMTarget)\installer\Release\$($Arch.VSName)\installer.exe" -engine "$Stage\installer-engine.exe" -intermediateFolder "$BinaryCache\$($Arch.LLVMTarget)\installer\$($Arch.VSName)\"
}

#-------------------------------------------------------------------
try {

Get-Dependencies

if ($Clean) {
  foreach ($project in [HostComponent]::GetNames([HostComponent])) {
    if ($project -eq "Compilers") { continue }
    Remove-Item -Force -Recurse -Path "$BinaryCache\$($HostArch.LLVMTarget)\$project" -ErrorAction Ignore
  }

  # In case of a previous test run, clear out the swiftmodules as they are not a stable format.
  Remove-Item -Force -Recurse -Path "$($HostArch.ToolchainInstallRoot)\usr\lib\swift\windows\*.swiftmodule" -ErrorAction Ignore
  foreach ($Arch in $WindowsSDKArchs) {
    foreach ($project in [TargetComponent]::GetNames([TargetComponent])) {
      Remove-Item -Force -Recurse -Path "$BinaryCache\$($Arch.LLVMTarget)\$project" -ErrorAction Ignore
    }
  }

  foreach ($Arch in $AndroidSDKArchs) {
    foreach ($project in [TargetComponent]::GetNames([TargetComponent])) {
      Remove-Item -Force -Recurse -Path "$BinaryCache\$($Arch.LLVMTarget)\$project" -ErrorAction Ignore
    }
  }

  Remove-Item -Force -Recurse ([IO.Path]::Combine((Get-InstallDir $HostArch), "Runtimes", $ProductVersion)) -ErrorAction Ignore
}

if (-not $SkipBuild) {
  if ($EnableCaching -And (-Not (Test-SCCacheAtLeast -Major 0 -Minor 7 -Patch 4))) {
    throw "Minimum required sccache version is 0.7.4"
  }

  Remove-Item -Force -Recurse ([IO.Path]::Combine((Get-InstallDir $HostArch), "Platforms")) -ErrorAction Ignore

  Invoke-BuildStep Build-CMark $BuildArch
  Invoke-BuildStep Build-BuildTools $BuildArch
  if ($IsCrossCompiling) {
    Invoke-BuildStep Build-XML2 Windows $BuildArch
    Invoke-BuildStep Build-Compilers -Build $BuildArch
  }
  if ($IncludeDS2) {
    Invoke-BuildStep Build-RegsGen2 $BuildArch
  }

  Invoke-BuildStep Build-CMark $HostArch
  Invoke-BuildStep Build-XML2 Windows $HostArch
  Invoke-BuildStep Build-Compilers $HostArch

  foreach ($Arch in $WindowsSDKArchs) {
    Invoke-BuildStep Build-ZLib Windows $Arch
    Invoke-BuildStep Build-XML2 Windows $Arch
    Invoke-BuildStep Build-CURL Windows $Arch
    Invoke-BuildStep Build-LLVM Windows $Arch

    # Build platform: SDK, Redist and XCTest
    Invoke-BuildStep Build-Runtime Windows $Arch
    Invoke-BuildStep Build-Dispatch Windows $Arch
    # FIXME(compnerd) ensure that the _build_ is the first arch and don't rebuild on each arch
    Invoke-BuildStep Build-FoundationMacros -Build Windows $BuildArch
    Invoke-BuildStep Build-TestingMacros -Build Windows $BuildArch
    Invoke-BuildStep Build-Foundation Windows $Arch
    Invoke-BuildStep Build-Sanitizers Windows $Arch
    Invoke-BuildStep Build-XCTest Windows $Arch
    Invoke-BuildStep Build-Testing Windows $Arch
    Invoke-BuildStep Write-SDKSettingsPlist Windows $Arch

    Invoke-BuildStep Build-ExperimentalRuntime -Static Windows $Arch
    Invoke-BuildStep Build-Foundation -Static Windows $Arch

    Copy-File "$(Get-SwiftSDK Windows)\usr\lib\swift\windows\*.lib" "$(Get-SwiftSDK Windows)\usr\lib\swift\windows\$($Arch.LLVMName)\"
    if ($Arch -eq $HostArch) {
      Copy-Directory "$(Get-SwiftSDK Windows)\usr\bin" "$([IO.Path]::Combine((Get-InstallDir $HostArch), "Runtimes", $ProductVersion))\usr"
    }
  }
  Install-Platform Windows $WindowsSDKArchs
  Invoke-BuildStep Write-PlatformInfoPlist Windows

  if ($Android) {
    foreach ($Arch in $AndroidSDKArchs) {
      if ($IncludeDS2) {
        Invoke-BuildStep Build-DS2 Android $Arch
      }
      Invoke-BuildStep Build-ZLib Android $Arch
      Invoke-BuildStep Build-XML2 Android $Arch
      Invoke-BuildStep Build-CURL Android $Arch
      Invoke-BuildStep Build-LLVM Android $Arch

      # Build platform: SDK, Redist and XCTest
      Invoke-BuildStep Build-Runtime Android $Arch
      Invoke-BuildStep Build-Dispatch Android $Arch
      Invoke-BuildStep Build-Foundation Android $Arch
      Invoke-BuildStep Build-Sanitizers Android $Arch
      Invoke-BuildStep Build-XCTest Android $Arch
      Invoke-BuildStep Build-Testing Android $Arch

      # Android swift-inspect only supports 64-bit platforms.
      if ($Arch.AndroidArchABI -eq "arm64-v8a" -or
          $Arch.AndroidArchABI -eq "x86_64") {
        Invoke-BuildStep Build-Inspect -Platform Android -Arch $Arch
      }
      Invoke-BuildStep Write-SDKSettingsPlist Android $Arch

      Invoke-BuildStep Build-ExperimentalRuntime -Static Android $Arch
      Invoke-BuildStep Build-Foundation -Static Android $Arch

      Move-Item "$(Get-SwiftSDK Android)\usr\lib\swift\android\*.a" "$(Get-SwiftSDK Android)\usr\lib\swift\android\$($Arch.LLVMName)\"
      Move-Item "$(Get-SwiftSDK Android)\usr\lib\swift\android\*.so" "$(Get-SwiftSDK Android)\usr\lib\swift\android\$($Arch.LLVMName)\"
    }
    Install-Platform Android $AndroidSDKArchs
    Invoke-BuildStep Write-PlatformInfoPlist Android
  }

  # Build Macros for distribution
  Invoke-BuildStep Build-FoundationMacros Windows $HostArch
  Invoke-BuildStep Build-TestingMacros Windows $HostArch

  Invoke-BuildStep Build-SQLite $HostArch
  Invoke-BuildStep Build-ToolsSupportCore $HostArch
  Invoke-BuildStep Build-LLBuild $HostArch
  Invoke-BuildStep Build-ArgumentParser $HostArch
  Invoke-BuildStep Build-Driver $HostArch
  Invoke-BuildStep Build-Crypto $HostArch
  Invoke-BuildStep Build-Collections $HostArch
  Invoke-BuildStep Build-ASN1 $HostArch
  Invoke-BuildStep Build-Certificates $HostArch
  Invoke-BuildStep Build-System $HostArch
  Invoke-BuildStep Build-Build $HostArch
  Invoke-BuildStep Build-PackageManager $HostArch
  Invoke-BuildStep Build-Markdown $HostArch
  Invoke-BuildStep Build-Format $HostArch
  Invoke-BuildStep Build-LMDB $HostArch
  Invoke-BuildStep Build-IndexStoreDB $HostArch
  Invoke-BuildStep Build-SourceKitLSP $HostArch
  Invoke-BuildStep Build-Inspect Windows $HostArch
}

Install-HostToolchain

if (-not $SkipBuild) {
  Invoke-BuildStep Build-mimalloc $HostArch
}

if (-not $SkipBuild -and -not $IsCrossCompiling) {
  Invoke-BuildStep Build-DocC $HostArch
}

if (-not $SkipPackaging) {
  Invoke-BuildStep Build-Installer $HostArch
}

if ($Stage) {
  Copy-BuildArtifactsToStage $HostArch
}

if (-not $IsCrossCompiling) {
  $CompilersTests = @("clang", "lld", "lldb", "llvm", "swift")
  if ($Test | Where-Object { $CompilersTests -contains $_ }) {
    $Tests = @{
      "-TestClang" = $Test -contains "clang";
      "-TestLLD" = $Test -contains "lld";
      "-TestLLDB" = $Test -contains "lldb";
      "-TestLLVM" = $Test -contains "llvm";
      "-TestSwift" = $Test -contains "swift";
    }
    Build-Compilers $HostArch @Tests
  }

  if ($Test -contains "dispatch") {
    Build-Dispatch Windows $HostArch -Test
  }
  if ($Test -contains "foundation") {
    Build-Foundation Windows $HostArch -Test
  }
  if ($Test -contains "xctest") {
    Test-XCTest
  }
  if ($Test -contains "testing") {
    Build-Testing Windows $HostArch -Test
  }
  if ($Test -contains "llbuild") { Build-LLBuild $HostArch -Test }
  if ($Test -contains "swiftpm") { Test-PackageManager $HostArch }
  if ($Test -contains "swift-format") { Test-Format }
  if ($Test -contains "sourcekit-lsp") { Test-SourceKitLSP }

  if ($Test -contains "swift") {
    foreach ($Arch in $AndroidSDKArchs) {
      Test-Runtime Android $Arch
    }
  }
}

# Custom exception printing for more detailed exception information
} catch {
  function Write-ErrorLines($Text, $Indent = 0) {
    $IndentString = " " * $Indent
    $Text.Replace("`r", "") -split "`n" | ForEach-Object {
      Write-Host "$IndentString$_" -ForegroundColor Red
    }
  }

  Write-ErrorLines "Error: $_"
  Write-ErrorLines $_.ScriptStackTrace -Indent 4

  # Walk the .NET inner exception chain to print all messages and stack traces
  $Exception = $_.Exception
  $Indent = 2
  while ($Exception -is [Exception]) {
      Write-ErrorLines "From $($Exception.GetType().FullName): $($Exception.Message)" -Indent $Indent
      if ($null -ne $Exception.StackTrace) {
          # .NET exceptions stack traces are already indented by 3 spaces
          Write-ErrorLines $Exception.StackTrace -Indent ($Indent + 1)
      }
      $Exception = $Exception.InnerException
      $Indent += 2
  }

  exit 1
} finally {
  if ($Summary) {
    Write-Summary
  }
}
