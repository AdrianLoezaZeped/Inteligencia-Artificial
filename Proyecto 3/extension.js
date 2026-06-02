const vscode = require("vscode");
const cp = require("child_process");

let proc;
let pending = [];
let outputBuffer = ""; // Manejo de flujo de datos para evitar JSON truncados

function request(method, payload) {
  return new Promise((resolve) => {
    pending.push(resolve);
    if (proc && proc.stdin) {
        proc.stdin.write(JSON.stringify({ method, ...payload }) + "\n");
    } else {
        vscode.window.showErrorMessage("Error: El motor de IA está apagado o no inicializó.");
        resolve({ok: false});
    }
  });
}

function activate(context) {
  const py = "py";
  const projectPath = "C:/Users/adria/Desktop/Carpetas/Escuela/IA/Proyecto 3";
  const script = projectPath + "/server_stdio.py";

  // Inicialización del proceso subyacente en modo desatendido (-u)
  proc = cp.spawn(py, ["-u", script], { cwd: projectPath });

  proc.stdout.on("data", (data) => {
    outputBuffer += data.toString();
    const lines = outputBuffer.split("\n");
    outputBuffer = lines.pop(); // Almacena fragmentos incompletos para el siguiente ciclo

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const res = JSON.parse(line);
        const cb = pending.shift();
        if (cb) cb(res);
      } catch (e) {
        console.error("Entrada ignorada por inconsistencia de formato:", line);
      }
    }
  });

  proc.stderr.on("data", (data) => {
      // Flujo de diagnóstico del sistema (TensorFlow / Keras Warnings)
      console.log("Motor RNN log:", data.toString());
  });

  context.subscriptions.push(
    vscode.commands.registerCommand("rnnKeras.complete", async () => {
      const ed = vscode.window.activeTextEditor;
      if (!ed) return;
      
      const pos = ed.selection.active;
      const prefix = ed.document.lineAt(pos.line).text.slice(0, pos.character);
      
      vscode.window.showInformationMessage("Generando bloque de código..."); 
      
      // Solicitud de predicción con balance de creatividad controlado (temp: 0.5)
      const res = await request("complete", { prefix, max_new: 200, temperature: 0.1 });
      
      if (res && res.ok) {
        let suffix = res.text.slice(prefix.length);
        
        // ========================================================
        // INTERCEPTOR DE SINTAXIS (Freno de la llave de cierre)
        // ========================================================
        const cierreIndex = suffix.indexOf('}');
        if (cierreIndex !== -1) {
            // Segmenta el código aislando la función y descartando remanentes
            suffix = suffix.slice(0, cierreIndex + 1);
        }

        await ed.edit((eb) => eb.insert(pos, suffix));
      }
    })
  );
}

function deactivate() {
  if (proc) proc.kill();
}

module.exports = { activate, deactivate };