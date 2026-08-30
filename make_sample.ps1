Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = -2
$outFile = Join-Path (Get-Location) "sample_speech.wav"
$synth.SetOutputToWaveFile($outFile)
$synth.Speak("Hello. This is a test of the meeting transcription tool. The quick brown fox jumps over the lazy dog. Testing one two three.")
$synth.SetOutputToDefaultAudioDevice()
$synth.Dispose()
Write-Host "Created: $outFile"
Get-Item $outFile | Select-Object Name, Length
