; Windows installer for the Hub Counter and Watchtower apps (Inno Setup 6).
; hub-app.yml builds one per app from PyInstaller's folder:
;   iscc /DAppName="Watchtower" /DAppExe="Watchtower.exe" /DSrcDir="C:\...\dist\Watchtower"
;        /DOutFile="Watchtower-Setup-windows" /DAppVersion=0.4.4 /O"C:\..." apps\installer\windows.iss
; SrcDir and the output folder absolute: Inno takes relative paths from this
; script's folder, not from where iscc runs.
;
; Why an installer and not the zip it replaces: a zip had to be extracted
; whole (the program needs its _internal folder beside it) and then run from
; wherever it landed, with no Start-menu entry; a double-clicked exe inside
; the zip ran without its folder and failed. This puts it in the user's
; programs folder with a Start-menu shortcut (desktop optional) and an
; uninstaller, per user, so school laptops without admin rights can install
; it. Not one self-extracting exe: with PyTorch inside, unpacking ~1 GB on
; every launch would make each start take a minute.

#ifndef AppName
  #define AppName "Watchtower"
#endif
#ifndef AppExe
  #define AppExe AppName + ".exe"
#endif
#ifndef SrcDir
  #define SrcDir AddBackslash("dist") + AppName
#endif
#ifndef OutFile
  #define OutFile AppName + "-Setup-windows"
#endif
#ifndef IconFile
  #define IconFile ""
#endif
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
; A fixed id per app, so a newer installer upgrades the old install in place.
; Not in braces: "{{#AppName}..." became "{Hub Counter-...}" after the
; preprocessor, which Inno then read as an unknown {constant}.
AppId={#AppName}-ShadowOfTheVOID-YOLOv26-FRC-Model
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=ShadowOfTheVOID / YOLOv26-FRC-Model
AppPublisherURL=https://github.com/ShadowOfTheVOID/YOLOv26-FRC-Model
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
; "x64", not 6.3's "x64compatible": the runner's preinstalled Inno may be older.
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
OutputDir=.
OutputBaseFilename={#OutFile}
Compression=lzma2/normal
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\{#AppExe}
; The installer's own icon (the app's), when hub-app.yml passes /DIconFile.
#if IconFile != ""
SetupIconFile={#IconFile}
#endif
; Close a running copy before replacing its files.
CloseApplications=yes

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "{#SrcDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Open {#AppName} now"; Flags: nowait postinstall skipifsilent
