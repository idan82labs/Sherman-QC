Attribute VB_Name = "SldprtToStep"
' SldprtToStep - export every .SLDPRT in a folder (and its subfolders) to STEP AP214 in mm.
' Sherman QC, 2026-10-28. How to run: README.md / INSTRUCTIONS_SHERMAN.md.
'
' - Asks for an input folder and an output folder (the output mirrors the subfolders).
' - Opens each part READ-ONLY and silently, never saves it, closes it after the export.
' - Sets the STEP export to AP214 and puts the user's own setting back at the end.
' - Units: the part's length unit is switched to mm in memory only (the part is never saved).
' - Writes <output>\manifest.csv: one line per part with status OK / SKIPPED_UP_TO_DATE / FAILED_OPEN /
'   FAILED_EXPORT / UNSUPPORTED_TYPE and the error text. Assemblies/drawings are listed, not exported.
' - A part whose STEP already exists and is newer than the part is skipped (SKIPPED_UP_TO_DATE).
' - Paths containing _SEALED are skipped and only counted.
' Needs the references SolidWorks adds to every new macro (SldWorks type library + constant type library).
Option Explicit

Dim swApp As SldWorks.SldWorks
Dim fso As Object
Dim csv As Object
Dim outRoot As String
Dim nOk As Long, nSkip As Long, nFail As Long, nOther As Long, nSealed As Long

Sub main()
    Set swApp = Application.SldWorks
    Set fso = CreateObject("Scripting.FileSystemObject")

    Dim inRoot As String
    inRoot = PickFolder("Folder with the .SLDPRT parts (subfolders included)")
    If inRoot = "" Then Exit Sub
    outRoot = PickFolder("Empty folder for the STEP files and manifest.csv")
    If outRoot = "" Then Exit Sub
    If LCase(inRoot) = LCase(outRoot) Then
        MsgBox "Pick a different output folder.", vbExclamation: Exit Sub
    End If

    Dim oldAP As Long
    oldAP = swApp.GetUserPreferenceIntegerValue(swStepAP)
    swApp.SetUserPreferenceIntegerValue swStepAP, 214

    On Error GoTo Cleanup
    Dim exists As Boolean
    exists = fso.FileExists(outRoot & "\manifest.csv")
    Set csv = fso.OpenTextFile(outRoot & "\manifest.csv", 8, True)   ' append
    If Not exists Then
        csv.WriteLine "source_rel,source_name,source_size,source_modified,configuration,output_rel,step_size,status,error,warnings,sw_revision,finished"
    End If
    WalkFolder fso.GetFolder(inRoot), inRoot

Cleanup:
    Dim errText As String
    If Err.Number <> 0 Then errText = "Stopped by an error: " & Err.Description & vbCrLf
    On Error Resume Next
    swApp.SetUserPreferenceIntegerValue swStepAP, oldAP
    If Not csv Is Nothing Then csv.Close
    MsgBox errText & "Done." & vbCrLf & "OK: " & nOk & vbCrLf & "Up to date (skipped): " & nSkip & vbCrLf & _
           "Failed: " & nFail & vbCrLf & "Assemblies/drawings (not exported): " & nOther & vbCrLf & _
           "Sealed paths skipped: " & nSealed & vbCrLf & vbCrLf & _
           "Send back: the whole folder " & outRoot & " zipped (STEP files + manifest.csv).", vbInformation
End Sub

Private Function PickFolder(title As String) As String
    Dim sh As Object, f As Object
    Set sh = CreateObject("Shell.Application")
    Set f = sh.BrowseForFolder(0, title, &H41, 0)   ' new-style dialog, edit box
    If f Is Nothing Then PickFolder = "" Else PickFolder = f.Self.Path
End Function

Private Sub WalkFolder(folder As Object, inRoot As String)
    Dim f As Object, sub1 As Object
    For Each f In folder.Files
        DoOneFile f, inRoot
    Next
    For Each sub1 In folder.SubFolders
        If InStr(1, sub1.Path, "_SEALED", vbTextCompare) = 0 Then
            WalkFolder sub1, inRoot
        Else
            nSealed = nSealed + 1
        End If
    Next
End Sub

Private Sub DoOneFile(f As Object, inRoot As String)
    Dim ext As String, rel As String
    ext = LCase(fso.GetExtensionName(f.Name))
    If ext <> "sldprt" And ext <> "sldasm" And ext <> "slddrw" Then Exit Sub
    If Left(f.Name, 2) = "~$" Then Exit Sub
    If InStr(1, f.Path, "_SEALED", vbTextCompare) > 0 Then nSealed = nSealed + 1: Exit Sub
    rel = Mid(f.Path, Len(inRoot) + 2)
    If ext <> "sldprt" Then
        nOther = nOther + 1
        WriteRow rel, f, "", "", "", "UNSUPPORTED_TYPE", "." & ext & " is not exported (parts only)", ""
        Exit Sub
    End If

    Dim outRel As String, outPath As String, tmpPath As String
    outRel = Left(rel, Len(rel) - Len(ext) - 1) & ".step"
    outPath = outRoot & "\" & outRel
    If fso.FileExists(outPath) Then
        If fso.GetFile(outPath).DateLastModified > f.DateLastModified Then
            nSkip = nSkip + 1
            WriteRow rel, f, "", outRel, CStr(fso.GetFile(outPath).Size), "SKIPPED_UP_TO_DATE", "", ""
            Exit Sub
        End If
    End If
    MakeFolders fso.GetParentFolderName(outPath)
    tmpPath = Left(outPath, Len(outPath) - 5) & ".~tmp.step"

    Dim doc As SldWorks.ModelDoc2
    Dim errs As Long, warns As Long, ok As Boolean, cfg As String, wtxt As String
    Set doc = swApp.OpenDoc6(f.Path, swDocPART, swOpenDocOptions_Silent + swOpenDocOptions_ReadOnly, "", errs, warns)
    If doc Is Nothing Then
        nFail = nFail + 1
        WriteRow rel, f, "", "", "", "FAILED_OPEN", OpenErrorText(errs), "warnings=" & warns
        Exit Sub
    End If
    If warns <> 0 Then wtxt = "open warnings=" & warns
    On Error Resume Next
    cfg = doc.ConfigurationManager.ActiveConfiguration.Name
    doc.ForceRebuild3 False
    If doc.LengthUnit <> swMM Then
        doc.Extension.SetUserPreferenceInteger swUnitSystem, swDetailingNoOptionSpecified, swUnitSystem_MMGS
        wtxt = wtxt & " unit " & doc.LengthUnit & "->mm"
    End If
    Err.Clear
    If fso.FileExists(tmpPath) Then fso.DeleteFile tmpPath, True
    Dim se As Long, sw2 As Long
    ok = doc.Extension.SaveAs(tmpPath, swSaveAsCurrentVersion, swSaveAsOptions_Silent + swSaveAsOptions_Copy, Nothing, se, sw2)
    Dim saveErr As String
    If Err.Number <> 0 Then saveErr = Err.Description: Err.Clear
    swApp.CloseDoc doc.GetTitle
    Set doc = Nothing
    If ok And fso.FileExists(tmpPath) Then
        If fso.GetFile(tmpPath).Size > 0 Then
            If fso.FileExists(outPath) Then fso.DeleteFile outPath, True
            fso.MoveFile tmpPath, outPath
        Else
            ok = False
        End If
    Else
        ok = False
    End If
    If ok And Err.Number = 0 Then
        nOk = nOk + 1
        WriteRow rel, f, cfg, outRel, CStr(fso.GetFile(outPath).Size), "OK", "", Trim(wtxt)
    Else
        nFail = nFail + 1
        If fso.FileExists(tmpPath) Then fso.DeleteFile tmpPath, True
        WriteRow rel, f, cfg, "", "", "FAILED_EXPORT", "SaveAs STEP failed: errors=" & se & " " & saveErr & " " & Err.Description, Trim(wtxt & " save warnings=" & sw2)
    End If
    Err.Clear
    On Error GoTo 0
End Sub

Private Function OpenErrorText(errs As Long) As String
    Dim t As String
    If errs And swFileNotFoundError Then t = t & "file not found; "
    If errs And swFutureVersion Then t = t & "saved in a newer SolidWorks version; "
    If errs And swInvalidFileTypeError Then t = t & "invalid file type; "
    If errs And swFileRequiresRepairError Then t = t & "file needs repair; "
    OpenErrorText = "open failed (errors=" & errs & ") " & t
End Function

Private Sub MakeFolders(p As String)
    If p = "" Or fso.FolderExists(p) Then Exit Sub
    MakeFolders fso.GetParentFolderName(p)
    fso.CreateFolder p
End Sub

Private Function Q(s As String) As String
    Q = """" & Replace(s, """", """""") & """"
End Function

Private Sub WriteRow(rel As String, f As Object, cfg As String, outRel As String, stepSize As String, _
                     status As String, errText As String, warnText As String)
    csv.WriteLine Q(rel) & "," & Q(f.Name) & "," & f.Size & "," & Q(Format(f.DateLastModified, "yyyy-mm-dd hh:nn:ss")) & "," & _
                  Q(cfg) & "," & Q(outRel) & "," & stepSize & "," & status & "," & Q(errText) & "," & Q(warnText) & "," & _
                  Q(swApp.RevisionNumber) & "," & Q(Format(Now, "yyyy-mm-dd hh:nn:ss"))
End Sub
