// Double-click launcher for the JaydariGPT desktop chat.
// Starts gui.py with the project's .venv (pythonw, so no console window).
// Build with launcher\build_launcher.bat.
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Windows.Forms;

[assembly: System.Reflection.AssemblyTitle("JaydariGPT")]
[assembly: System.Reflection.AssemblyProduct("JaydariGPT")]
[assembly: System.Reflection.AssemblyDescription("Private local AI chat")]
[assembly: System.Reflection.AssemblyVersion("1.0.0.0")]

static class Launcher
{
    const string Title = "JaydariGPT";

    [STAThread]
    static int Main()
    {
        string dir = AppDomain.CurrentDomain.BaseDirectory;
        string pythonw = Path.Combine(dir, @".venv\Scripts\pythonw.exe");
        string gui = Path.Combine(dir, "gui.py");
        string baseModel = Path.Combine(dir, @"models\TinyLlama-1.1B-Chat-v1.0");
        string log = Path.Combine(dir, "jaydarigpt.log");

        if (!File.Exists(gui))
            return Fail("gui.py was not found next to JaydariGPT.exe.\n\n" +
                        "Keep JaydariGPT.exe inside the JaydariGPT_Local folder.\n" +
                        "To start it from the Desktop, create a shortcut instead of moving it.");
        if (!File.Exists(pythonw))
            return Fail("JaydariGPT is not installed yet.\n\n" +
                        "1. Double-click INSTALL_WINDOWS.bat (needs internet).\n" +
                        "2. Then open JaydariGPT.exe again.");
        if (!Directory.Exists(baseModel))
            return Fail("The base model has not been downloaded yet.\n\n" +
                        "While online, run in this folder:\n" +
                        ".venv\\Scripts\\python.exe download_base_model.py\n\n" +
                        "Then open JaydariGPT.exe again.");

        Process app;
        try
        {
            app = Process.Start(new ProcessStartInfo(pythonw, "\"" + gui + "\"")
            {
                WorkingDirectory = dir,
                UseShellExecute = false,
            });
        }
        catch (Exception ex)
        {
            return Fail("Could not start Python:\n\n" + ex.Message);
        }

        // A crash during the first seconds (broken install) would otherwise be silent.
        if (app.WaitForExit(6000) && app.ExitCode != 0 && File.Exists(log))
        {
            string[] lines = ReadShared(log);
            string tail = string.Join("\n", lines.Skip(Math.Max(0, lines.Length - 12)));
            if (tail.Trim().Length > 0)
                return Fail("JaydariGPT closed unexpectedly.\n\n" + tail + "\n\nFull log: " + log);
        }
        return 0;
    }

    static string[] ReadShared(string path)
    {
        using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
        using (var reader = new StreamReader(stream))
            return reader.ReadToEnd().Split('\n');
    }

    static int Fail(string message)
    {
        MessageBox.Show(message, Title, MessageBoxButtons.OK, MessageBoxIcon.Warning);
        return 1;
    }
}
