import { app } from "../../scripts/app.js";
import { ComfyButtonGroup } from "../../scripts/ui/components/buttonGroup.js";
import { ComfyButton } from "../../scripts/ui/components/button.js";

async function jinnExportWorkflowToPhotoshop(e) {

    try {
        const wf = await app.graphToPrompt();
        if(!wf ||!wf.output){alert('❌当前没有打开工作流，或者无法正确获取到工作流');return;}
        const resp = await fetch('/jinn_save_wf_new_byui', {method: 'POST',body:JSON.stringify({wf:wf})})
        if (!resp.ok) {alert(`❌访问服务端出错${resp.status} ${resp.statusText}`);return;}
        const data = await resp.json();
        if ("comfy_error" in data) {alert(`❌发生错误：${data.comfy_error}`);return;}
        alert(`✔️${data.result},请在PS插件上点击刷新按钮`);
    } catch (error) {
        console.error( error);
        alert(`❌${error.message}`);
    }
}
function addTopBarButtons() {
    let jinnButtonGroup = null;
    let _b, _c;
    if (jinnButtonGroup) {
        (_b = app.menu) === null || _b === void 0 ? void 0 : _b.settingsGroup.element.before(jinnButtonGroup.element);
        return;
    }
    const buttons = [];
    const btAPI = new ComfyButton({ content: '添加到PS', tooltip: "将工作流添加到PS中", app, action: jinnExportWorkflowToPhotoshop });
    buttons.push(btAPI);
    jinnButtonGroup = new ComfyButtonGroup(...buttons);
    (_c = app.menu) === null || _c === void 0 ? void 0 : _c.settingsGroup.element.before(jinnButtonGroup.element);
}
app.registerExtension({
    name: "jinn.TopMenu",
    async setup() {
        //addTopBarButtons();//导出的API不正确，暂时弃用
    },
});
