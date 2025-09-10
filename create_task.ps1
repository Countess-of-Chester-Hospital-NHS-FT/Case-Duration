# Define variables
$TaskName = "Run Case Duration Prediction Script"
$Description = "Schedules the case duration script to run twice daily."
$PythonPath = "C:\Users\Helena Robinson\.conda\envs\case-duration\python.exe" # Replace with your actual Python path
$ScriptPath = "C:\Users\Helena Robinson\Desktop\Case-Duration\predict.py" # Replace with your actual script path
$WorkingDirectory = "C:\Users\Helena Robinson\Desktop\Case-Duration" # The directory where your script is located

# 1. Create an Action: What to run
$Action = New-ScheduledTaskAction -Execute $PythonPath -Argument "`"$ScriptPath`"" -WorkingDirectory $WorkingDirectory

# 2. Create a Trigger: When to run (e.g., daily at 10:00 AM)
#$Trigger = New-ScheduledTaskTrigger -Daily -At "01:06 PM"
$Trigger1 = New-ScheduledTaskTrigger -Daily -At "09:29 AM"
$Trigger2 = New-ScheduledTaskTrigger -Daily -At "10:29 AM"
$Triggers = @($Trigger1, $Trigger2)

# You can also use other triggers, for example:
# $Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5) # Run once, 5 minutes from now
# $Trigger = New-ScheduledTaskTrigger -AtLogon # Run at user logon
# $Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Wednesday,Friday -At "09:00 AM"

# 3. Create Task Settings: Additional configurations
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RunOnlyIfNetworkAvailable:$false

# 4. Register the Scheduled Task
# For running under system
#Register-ScheduledTask -TaskName $TaskName -Description $Description -Action $Action -Trigger $Trigger -Settings $Settings -User "System" -Force
# For running under the current user:
Register-ScheduledTask -TaskName $TaskName -Description $Description -Action $Action -Trigger $Triggers -Settings $Settings -User $env:UserName -Force

Write-Host "Scheduled task '$TaskName' created successfully!"