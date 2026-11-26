##include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include <Preferences.h>
#include <math.h>
#include "ScioSense_ENS160.h"
#include <Adafruit_AHTX0.h>

const char* SSID = "Firefox";
const char* PASS = "87654321";

WebServer server(80);
Preferences prefs;
ScioSense_ENS160 ens160(0x53); // Endereço I2C real da placa azul
Adafruit_AHTX0 aht;            // Endereço I2C 0x38

const int PIN_MQ2 = 32, PIN_MQ135 = 33, PIN_MQ7 = 34;
const int PIN_LED_G = 25, PIN_LED_R = 27;
const int PIN_BUZ1 = 14, PIN_BUZ2 = 12;

bool ahtOk = false, ensOk = false, ensWarm = false;
bool mq2Ok = false, mq135Ok = false, mq7Ok = false;

float base2 = 0, base135 = 0, base7 = 0;
float filt2 = 0, filt135 = 0, filt7 = 0;

int ppm2 = 0, ppm135 = 0, ppm7 = 0;
int pct2 = 0, pct135 = 0, pct7 = 0, pctTvoc = 0, pctEco2 = 0, pctTemp = 0;
float tempC = 0, umidPct = 0;
uint8_t aqi = 0;
uint16_t tvoc = 0, eco2 = 400;

int contAtencao = 0, contGrave = 0;
bool atencao = false, grave = false;
int assinaturaId = 0;
int confiancaMLP = 100;
bool modoTreinoAtivo = false;

int volBuzzerPct = 15; // Volume padrão (0% a 100%)
unsigned long tLeitura = 0, tWifi = 0, tBuz = 0, tTesteBuz = 0;
int passoBuz = 0;

// ===================== PESOS INICIAIS DA REDE NEURAL MLP (5 -> 8 -> 7) =====================
// Entradas exclusivas de gases: [0]=TVOC, [1]=eCO2, [2]=MQ-2, [3]=MQ-135, [4]=MQ-7
const float W1_INIT[8][5] = {
  {  2.2f,  1.8f, -1.5f,  0.2f, -2.5f }, // 0: Álcool / Solvente Orgânico
  { -1.2f, -0.5f, -1.5f,  3.2f, -1.5f }, // 1: Amônia (NH3)
  { -1.5f, -0.5f,  3.2f, -1.2f, -1.2f }, // 2: GLP / Gás Inflamável
  { -1.5f, -0.5f, -1.2f, -1.2f,  3.2f }, // 3: Monóxido de Carbono (CO)
  {  1.2f,  0.8f,  1.4f,  1.4f,  1.4f }, // 4: Fumaça / Combustão
  { -1.5f,  3.2f, -1.5f, -1.5f, -1.5f }, // 5: Acúmulo de CO2
  {  2.0f, -1.0f, -1.8f,  2.2f, -1.8f }, // 6: Vapores Ácidos / NOx
  {  0.6f,  0.6f,  0.6f,  0.6f,  0.6f }  // 7: Geral
};
const float BIAS1_INIT[8] = { -0.8f, -0.8f, -0.8f, -0.8f, -2.6f, -0.8f, -1.2f, -0.2f };

const float W2_INIT[7][8] = {
  { 2.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.3f },
  { 0.0f, 2.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.3f },
  { 0.0f, 0.0f, 2.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.3f },
  { 0.0f, 0.0f, 0.0f, 2.2f, 0.0f, 0.0f, 0.0f, 0.3f },
  { 0.0f, 0.0f, 0.0f, 0.0f, 2.2f, 0.0f, 0.0f, 0.2f },
  { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.2f, 0.0f, 0.3f },
  { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.2f, 0.2f }
};
const float BIAS2_INIT[7] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

// Protótipos âncora para evitar esquecimento catastrófico durante o treino online
const float ANCORAS[7][5] = {
  {1.00f, 0.90f, 0.25f, 0.25f, 0.06f}, // 1: Álcool
  {0.20f, 0.15f, 0.10f, 1.00f, 0.10f}, // 2: Amônia
  {0.10f, 0.10f, 1.00f, 0.20f, 0.12f}, // 3: GLP
  {0.15f, 0.15f, 0.15f, 0.15f, 1.00f}, // 4: CO
  {0.90f, 0.85f, 0.95f, 0.85f, 0.80f}, // 5: Fumaça
  {0.35f, 1.00f, 0.05f, 0.05f, 0.05f}, // 6: CO2
  {0.95f, 0.50f, 0.05f, 0.80f, 0.05f}  // 7: Ácidos/NOx
};

float W1[8][5];
float bias1[8];
float W2[7][8];
float bias2[7];
bool iaPersonalizada = false;

// ===================== RELATÓRIOS SALVOS NA MEMÓRIA NÃO VOLÁTIL =====================
#define MAX_REL 20
struct RelatorioTreino {
  uint32_t id;
  uint32_t tempoSeg;
  uint8_t classeId;
  uint8_t confAntes;
  uint8_t confDepois;
  uint8_t pTvoc;
  uint8_t pEco2;
  uint8_t pMq2;
  uint8_t pMq135;
  uint8_t pMq7;
  uint16_t rawTvoc;
  uint16_t rawEco2;
  int16_t ppm2;
  int16_t ppm135;
  int16_t ppm7;
  char nota[24];
};

RelatorioTreino relatorios[MAX_REL];
uint8_t qtdRelatorios = 0;
uint32_t seqRelatorio = 0;

const char* NOMES_CLASSES[8] = {
  "Ambiente Limpo",
  "Álcool / Solventes",
  "Amônia (NH3)",
  "GLP / Gás Inflamável",
  "Monóxido de Carbono (CO)",
  "Fumaça / Combustão",
  "Acúmulo de CO2",
  "Vapores Ácidos / NOx"
};

const char HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html><html lang="pt-BR"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Monitor Lab Química</title>
<style>
body{font-family:Arial,sans-serif;background:#f1f5f9;color:#0f172a;margin:0;padding:12px}
.box{max-width:980px;margin:auto}
.top{background:#0f172a;color:#fff;padding:14px;border-radius:10px;margin-bottom:10px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}
.top h2{margin:0;font-size:18px}.top small{color:#94a3b8}
.tabs{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap}
.tab{padding:10px 16px;border-radius:8px;border:1px solid #cbd5e1;background:#fff;font-weight:bold;cursor:pointer;font-size:13px;color:#334155}
.tab.act{background:#0284c7;color:#fff;border-color:#0284c7}
.ban{padding:12px;border-radius:10px;font-weight:bold;margin-bottom:12px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px}
.ok{background:#dcfce7;color:#14532d}.at{background:#fef3c7;color:#78350f}.gr{background:#dc2626;color:#fff}
.card{background:#fff;padding:14px;border-radius:10px;margin-bottom:12px;border:1px solid #e2e8f0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(195px,1fr));gap:8px}
.item{background:#f8fafc;padding:10px 12px;border-radius:8px;border:1px solid #e2e8f0;display:flex;justify-content:space-between;align-items:center;gap:8px}
.pill{padding:4px 12px;border-radius:999px;color:#fff;font-size:12px;font-weight:bold;white-space:nowrap}
.on{background:#16a34a}.off{background:#dc2626}
.sig{padding:12px;border-radius:8px;background:#eff6ff;border-left:4px solid #0284c7;font-size:14px}
.sig.at{background:#fef3c7;border-left-color:#d97706}
.sig.gr{background:#fee2e2;border-left-color:#dc2626;color:#7f1d1d}
.conf-tag{display:inline-block;background:rgba(15,23,42,0.08);padding:2px 8px;border-radius:6px;font-size:12px;font-weight:bold;margin-left:6px}
.sub-list{font-size:13px;color:#334155;margin-top:6px;line-height:1.5}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:8px 6px;border-bottom:1px solid #e2e8f0;text-align:left}
.bar{background:#e2e8f0;height:9px;border-radius:9px;overflow:hidden;width:90px;display:inline-block;vertical-align:middle;margin-right:6px}
.fill{height:100%}
canvas{width:100%;height:180px;display:block}
.btn{padding:10px 14px;border:none;border-radius:8px;font-weight:bold;cursor:pointer;font-size:13px;color:#fff}
.b-sm{padding:6px 11px;font-size:12px;border-radius:6px}
.b-blue{background:#0284c7}.b-red{background:#dc2626}.b-green{background:#16a34a}.b-gray{background:#475569}
.btn:disabled{opacity:.5;cursor:not-allowed}
.tr-box{background:#f8fafc;border:1px solid #cbd5e1;padding:12px;border-radius:8px;margin-top:10px}
.pk-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin:10px 0}
.pk{background:#fff;border:1px solid #e2e8f0;padding:8px;border-radius:6px;text-align:center;font-size:12px}
.pk b{display:block;font-size:15px;color:#0f172a;margin-top:2px}
select,input[type=text]{padding:9px;border-radius:6px;border:1px solid #cbd5e1;font-size:13px;width:100%;box-sizing:border-box;margin-top:4px;margin-bottom:10px}
.msg{padding:10px;border-radius:8px;background:#dcfce7;color:#14532d;font-weight:bold;margin-top:10px;display:none}
.msg.err{background:#fee2e2;color:#7f1d1d}
</style></head><body><div class="box">
<div class="top">
  <div><h2>Laboratório de Química — Monitor Inteligente de Gases</h2><small id="ip">Conectando...</small></div>
  <small id="ia-st" style="color:#38bdf8;font-weight:bold">IA: Modelo Padrão</small>
</div>

<div class="tabs">
  <button id="t-mon" class="tab act" onclick="setTab('mon')">Monitoramento em Tempo Real</button>
  <button id="t-trn" class="tab" onclick="setTab('trn')">Modo Treinamento &amp; Relatórios</button>
</div>

<div id="ban" class="ban ok"><span id="btxt">AMBIENTE SEGURO</span><span id="qar">Qualidade do Ar: Avaliando...</span></div>

<div class="card"><b>Status dos Sensores</b><div class="grid" style="margin-top:8px">
<div class="item"><span>ENS160 + AHT2X</span><span id="s-combo" class="pill off">Desativado</span></div>
<div class="item"><span>MQ-2</span><span id="s-m2" class="pill off">Desativado</span></div>
<div class="item"><span>MQ-135</span><span id="s-m135" class="pill off">Desativado</span></div>
<div class="item"><span>MQ-7</span><span id="s-m7" class="pill off">Desativado</span></div>
</div></div>

<div class="card" style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px">
  <div>
    <b>Volume dos Buzzers (Alarme Grave): <span id="vol-lbl" style="color:#0284c7">15%</span></b>
    <div style="font-size:12px;color:#64748b">Salvo na memória do ESP32 (0% = mudo, 100% = volume máximo)</div>
  </div>
  <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
    <button class="btn b-gray b-sm" type="button" onclick="stepVol(-10)">−</button>
    <input type="range" id="vol-rng" min="0" max="100" step="5" value="15" style="width:140px;cursor:pointer"
      oninput="updVolLbl(this.value)" onchange="setVol(this.value,1)">
    <button class="btn b-gray b-sm" type="button" onclick="stepVol(10)">+</button>
    <button class="btn b-blue b-sm" type="button" onclick="testBuz()">🔊 Testar Som</button>
  </div>
</div>

<!-- ABA MONITORAMENTO -->
<div id="view-mon">
  <div class="card"><b>Diagnóstico provável</b>
  <div id="als" style="margin-top:8px">Executando inferência neural...</div></div>

  <div class="card"><b>Leituras Individuais (ppm e Escala 0% a 100%)</b>
  <table><thead><tr><th>Sensor</th><th>Parâmetro / Faixa</th><th>Valor</th><th>Escala (0-100%)</th></tr></thead>
  <tbody id="tb"></tbody></table></div>

  <div class="card"><b>Gráfico em Tempo Real (0% a 100%)</b>
  <div style="font-size:12px;margin:6px 0">
  <span style="color:#dc2626">■ CO (MQ-7)</span> &nbsp;
  <span style="color:#7c3aed">■ Poluentes (MQ-135)</span> &nbsp;
  <span style="color:#ea580c">■ Inflamável (MQ-2)</span> &nbsp;
  <span style="color:#0284c7">■ TVOC/eCO₂</span> &nbsp;
  <span style="color:#0d9488">■ Umidade</span>
  </div><canvas id="cv"></canvas></div>
</div>

<!-- ABA MODO TREINAMENTO -->
<div id="view-trn" style="display:none">
  <div class="card">
    <b>1. Simulação e Realimentação da Rede Neural MLP</b>
    <p style="font-size:13px;color:#475569;margin:6px 0">
      Clique em <b>Iniciar Simulação</b>, aproxime a substância dos sensores e clique em <b>Finalizar Captura e Identificar</b>. A rede neural considera apenas os 5 canais de gases (TVOC, eCO₂, MQ-2, MQ-135 e MQ-7) — temperatura e umidade não entram no cálculo da IA.
    </p>
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px">
      <button id="btn-cap" class="btn b-blue" onclick="toggleCap()">▶ Iniciar Simulação (Capturar Picos)</button>
      <button class="btn b-gray" onclick="resetIA()">Restaurar Pesos Padrão da IA</button>
    </div>

    <div id="cap-panel" class="tr-box" style="display:none">
      <b id="cap-tit" style="color:#0284c7">● Gravando picos dos sensores... Aproxime a substância agora!</b>
      <div class="pk-grid">
        <div class="pk">TVOC (ENS160)<b id="pk-tv">0% (0 ppb)</b></div>
        <div class="pk">eCO₂ (ENS160)<b id="pk-ec">0% (400 ppm)</b></div>
        <div class="pk">MQ-2 (Inflamável)<b id="pk-m2">0% (0 ppm)</b></div>
        <div class="pk">MQ-135 (Poluentes)<b id="pk-m135">0% (0 ppm)</b></div>
        <div class="pk">MQ-7 (CO)<b id="pk-m7">0% (0 ppm)</b></div>
      </div>

      <div id="form-rotulo" style="display:none;border-top:1px solid #cbd5e1;padding-top:10px;margin-top:8px">
        <label style="font-size:13px;font-weight:bold">Qual substância foi testada nesta simulação?</label>
        <select id="sel-cls">
          <option value="1">1 — Vapor de Álcool / Solvente Orgânico (Etanol, Gel, Acetona)</option>
          <option value="2">2 — Amônia (NH₃) / Vapores Alcalinos</option>
          <option value="3">3 — GLP / Gás Inflamável (Propano, Butano, Metano)</option>
          <option value="4">4 — Monóxido de Carbono (CO)</option>
          <option value="5">5 — Fumaça / Combustão (Fósforo, Incenso)</option>
          <option value="6">6 — Acúmulo de CO₂</option>
          <option value="7">7 — Vapores Ácidos / Oxidantes (NOₓ)</option>
        </select>
        <label style="font-size:13px;font-weight:bold">Observação do teste (opcional, até 20 caracteres):</label>
        <input type="text" id="inp-nota" maxlength="20" placeholder="Ex: Fosforo, Alcool 70, Amonia...">
        <button class="btn b-green" onclick="enviarTreino()">✅ Treinar Rede Neural e Salvar Relatório</button>
      </div>
    </div>
    <div id="tr-msg" class="msg"></div>
  </div>

  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:8px">
      <b>2. Relatórios Salvos na Memória Não Volátil (NVS Flash)</b>
      <div style="display:flex;gap:8px">
        <a href="/relatorios.csv" download="relatorio_lab_gases.csv"><button class="btn b-blue" type="button">📥 Baixar Relatório (.CSV)</button></a>
        <button class="btn b-red" type="button" onclick="limparRel()">Limpar Todos</button>
      </div>
    </div>
    <div style="overflow-x:auto">
      <table>
        <thead><tr><th>#</th><th>Tempo</th><th>Substância Informada</th><th>Obs.</th><th>Picos Analisados (TVOC | eCO₂ | MQ2 | MQ135 | MQ7)</th><th>Conf. IA</th><th>Ação</th></tr></thead>
        <tbody id="tb-rel"><tr><td colspan="7">Carregando relatórios...</td></tr></tbody>
      </table>
    </div>
  </div>
</div>

</div>
<script>
const H={co:[],nh:[],lp:[],vo:[],hu:[]};
const AQI_TXT=['Estabilizando (0%)','Excelente (1/5)','Boa (2/5)','Moderada (3/5)','Ruim (4/5)','Muito Ruim (5/5)'];
const SIG_INFO=[
  ['Ambiente Limpo','Nenhuma substância nociva ou alteração relevante detectada no ar do laboratório.'],
  ['Vapor de Álcool / Solvente Orgânico (Etanol, Gel, Acetona)','A rede neural identificou forte presença de compostos orgânicos (TVOC/eCO₂) com baixa resposta de CO (MQ-7), descartando falso alarme.'],
  ['Amônia (NH₃) ou Vapores Alcalinos','Ativação dominante no sensor MQ-135 com baixa resposta em gases combustíveis (MQ-2) e monóxido de carbono (MQ-7).'],
  ['Vazamento de GLP / Gás Inflamável (Propano, Butano, Metano)','Ativação dominante no sensor MQ-2 sem aumento proporcional de TVOC no ENS160 (padrão típico de gás combustível).'],
  ['Monóxido de Carbono (CO) — Combustão Incompleta','Ativação dominante no sensor MQ-7 sem presença proporcional de vapores orgânicos (TVOC). Ventile o laboratório imediatamente!'],
  ['Fumaça ou Gases de Queima','Ativação simultânea nos sensores MQ-2, MQ-135, MQ-7 e ENS160 característica de fumaça e combustão.'],
  ['Acúmulo de CO₂ (Ventilação Insuficiente)','Elevação concentrada de CO₂ equivalente no ENS160 sem presença de gases tóxicos ou inflamáveis nos sensores MQ.'],
  ['Vapores Ácidos ou Oxidantes (NOₓ / Reagentes)','Assinatura combinada no ENS160 e MQ-135 com ausência de gases combustíveis no MQ-2.']
];

let capturando=false;
let draggingVol=false;
let pico={pt:0,pe:0,p2:0,p135:0,p7:0,rt:0,re:400,ppm2:0,ppm135:0,ppm7:0};

function updVolLbl(v){
  draggingVol=true;
  document.getElementById('vol-lbl').textContent=v+'%';
}
async function setVol(v,beep){
  v=Math.min(100,Math.max(0,parseInt(v)||0));
  document.getElementById('vol-rng').value=v;
  document.getElementById('vol-lbl').textContent=v+'%';
  await fetch(`/volume?v=${v}&beep=${beep?1:0}`);
  draggingVol=false;
}
function stepVol(d){
  const cur=parseInt(document.getElementById('vol-rng').value)||0;
  setVol(cur+d,1);
}
async function testBuz(){
  const cur=parseInt(document.getElementById('vol-rng').value)||0;
  await fetch(`/volume?v=${cur}&beep=1`);
}

function setTab(t){
  document.getElementById('view-mon').style.display=(t==='mon')?'block':'none';
  document.getElementById('view-trn').style.display=(t==='trn')?'block':'none';
  document.getElementById('t-mon').className='tab'+(t==='mon'?' act':'');
  document.getElementById('t-trn').className='tab'+(t==='trn'?' act':'');
  fetch('/modo?m='+(t==='trn'?1:0));
  if(t==='trn') loadRelatorios();
}

function toggleCap(){
  const btn=document.getElementById('btn-cap');
  if(!capturando){
    capturando=true;
    pico={pt:0,pe:0,p2:0,p135:0,p7:0,rt:0,re:400,ppm2:0,ppm135:0,ppm7:0};
    atualizaPicoUI();
    document.getElementById('cap-panel').style.display='block';
    document.getElementById('form-rotulo').style.display='none';
    document.getElementById('tr-msg').style.display='none';
    document.getElementById('cap-tit').textContent='● Gravando picos dos sensores... Aproxime a substância agora!';
    document.getElementById('cap-tit').style.color='#dc2626';
    btn.textContent='⏹ Finalizar Captura e Identificar';
    btn.className='btn b-red';
  } else {
    capturando=false;
    document.getElementById('cap-tit').textContent='✔ Captura concluída! Selecione abaixo qual substância foi testada:';
    document.getElementById('cap-tit').style.color='#16a34a';
    document.getElementById('form-rotulo').style.display='block';
    btn.textContent='▶ Iniciar Nova Simulação';
    btn.className='btn b-blue';
  }
}

function atualizaPicoUI(){
  document.getElementById('pk-tv').textContent=`${pico.pt}% (${pico.rt} ppb)`;
  document.getElementById('pk-ec').textContent=`${pico.pe}% (${pico.re} ppm)`;
  document.getElementById('pk-m2').textContent=`${pico.p2}% (${pico.ppm2} ppm)`;
  document.getElementById('pk-m135').textContent=`${pico.p135}% (${pico.ppm135} ppm)`;
  document.getElementById('pk-m7').textContent=`${pico.p7}% (${pico.ppm7} ppm)`;
}

async function enviarTreino(){
  const cls=document.getElementById('sel-cls').value;
  const nota=encodeURIComponent(document.getElementById('inp-nota').value.trim());
  const url=`/treinar?cls=${cls}&nota=${nota}&pt=${pico.pt}&pe=${pico.pe}&p2=${pico.p2}&p135=${pico.p135}&p7=${pico.p7}&rt=${pico.rt}&re=${pico.re}&ppm2=${pico.ppm2}&ppm135=${pico.ppm135}&ppm7=${pico.ppm7}`;
  const res=await(await fetch(url)).json();
  const m=document.getElementById('tr-msg');
  m.style.display='block';
  if(res.erro){
    m.className='msg err';
    m.textContent=res.erro;
    return;
  }
  m.className='msg';
  m.textContent=`Rede Neural treinada e salva na Flash! Confiança para "${res.nome}": ${res.antes}% → ${res.depois}% (Relatório #${res.id} gravado).`;
  document.getElementById('form-rotulo').style.display='none';
  document.getElementById('inp-nota').value='';
  loadRelatorios();
  up();
}

async function excluirRel(id){
  if(!confirm(`Excluir o relatório #${id}? A Rede Neural será recalculada automaticamente desconsiderando esse teste.`))return;
  await fetch('/excluir_rel?id='+id);
  const m=document.getElementById('tr-msg');
  m.style.display='block';
  m.className='msg';
  m.textContent=`Relatório #${id} excluído e Rede Neural recalculada apenas com os testes válidos restantes.`;
  loadRelatorios();
  up();
}

async function loadRelatorios(){
  const list=await(await fetch('/relatorios')).json();
  const tb=document.getElementById('tb-rel');
  if(!list.length){tb.innerHTML='<tr><td colspan="7">Nenhum relatório salvo na memória não volátil.</td></tr>';return;}
  tb.innerHTML=list.slice().reverse().map(r=>`<tr>
    <td><b>#${r.id}</b></td>
    <td>${r.t}s</td>
    <td><b>${r.nome}</b></td>
    <td>${r.nota||'—'}</td>
    <td>TV:${r.pt}% | eCO₂:${r.pe}% | MQ2:${r.p2}% | MQ135:${r.p135}% | MQ7:${r.p7}%</td>
    <td>${r.ca}% → <b>${r.cd}%</b></td>
    <td><button class="btn b-red b-sm" type="button" onclick="excluirRel(${r.id})">Excluir</button></td>
  </tr>`).join('');
}

async function resetIA(){
  if(!confirm('Restaurar os pesos originais de fábrica da Rede Neural?'))return;
  await fetch('/reset_ia');
  up();
  alert('Pesos originais restaurados!');
}

async function limparRel(){
  if(!confirm('Apagar TODOS os relatórios e restaurar a IA para o modelo padrão?'))return;
  await fetch('/reset_rel');
  loadRelatorios();
  up();
}

function pill(id,ok){const e=document.getElementById(id);e.className='pill '+(ok?'on':'off');e.textContent=ok?'Ativado':'Desativado';}
function row(s,p,v,pc,ok,cl){
 const c=!ok?'#94a3b8':(cl?'#0284c7':(pc>=70?'#dc2626':(pc>=36?'#d97706':'#16a34a')));
 const w=ok?Math.min(100,Math.max(0,pc)):0;
 return `<tr><td><b>${s}</b></td><td>${p}</td><td>${ok?v:'—'}</td><td><div class="bar"><div class="fill" style="width:${w}%;background:${c}"></div></div><b>${ok?w+'%':'—'}</b></td></tr>`;
}
function draw(){
 const c=document.getElementById('cv'),ctx=c.getContext('2d');
 c.width=c.clientWidth;c.height=180;const W=c.width,Y=180;
 ctx.clearRect(0,0,W,Y);ctx.strokeStyle='#e2e8f0';
 [0,35,70,100].forEach(v=>{const y=Y-(v/100)*(Y-20)-10;ctx.beginPath();ctx.moveTo(28,y);ctx.lineTo(W,y);ctx.stroke();ctx.fillStyle='#64748b';ctx.fillText(v+'%',2,y+3);});
 const ser=[[H.hu,'#0d9488'],[H.vo,'#0284c7'],[H.lp,'#ea580c'],[H.nh,'#7c3aed'],[H.co,'#dc2626']];
 ser.forEach(([a,col])=>{if(a.length<2)return;ctx.strokeStyle=col;ctx.lineWidth=2;ctx.beginPath();
 a.forEach((v,i)=>{const x=28+(i/29)*(W-34),y=Y-(v/100)*(Y-20)-10;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();});
}
async function up(){
 try{
  const d=await(await fetch('/dados')).json();
  document.getElementById('ip').textContent='Wi-Fi: Firefox | IP: '+d.ip;
  document.getElementById('ia-st').textContent=d.iap?'IA: Treinada (NVS)':'IA: Modelo Padrão';
  if(!draggingVol && d.vol!==undefined){
    document.getElementById('vol-rng').value=d.vol;
    document.getElementById('vol-lbl').textContent=d.vol+'%';
  }
  const ensOn=d.ens[0],ahtOn=d.aht[0],aqiVal=d.ens[2];

  if(capturando){
    if(ensOn&&!d.ens[1]){
      if(d.ens[5]>pico.pt){pico.pt=d.ens[5];pico.rt=d.ens[3];}
      if(d.ens[6]>pico.pe){pico.pe=d.ens[6];pico.re=d.ens[4];}
    }
    if(d.m2[0]&&d.m2[2]>pico.p2){pico.p2=d.m2[2];pico.ppm2=d.m2[1];}
    if(d.m135[0]&&d.m135[2]>pico.p135){pico.p135=d.m135[2];pico.ppm135=d.m135[1];}
    if(d.m7[0]&&d.m7[2]>pico.p7){pico.p7=d.m7[2];pico.ppm7=d.m7[1];}
    atualizaPicoUI();
  }

  let qLabel='Sem sensores ativos';
  if(ensOn) qLabel=d.ens[1]?'Estabilizando sensor...':(AQI_TXT[aqiVal]||'Boa (2/5)');
  else if(d.m135[0]||d.m2[0]||d.m7[0]) qLabel=d.gr?'Muito Ruim (Crítica)':(d.at?'Moderada / Alterada':'Boa (Ar Limpo)');
  document.getElementById('qar').textContent='Qualidade do Ar: '+qLabel;

  const b=document.getElementById('ban');
  b.className='ban '+(d.gr?'gr':(d.at?'at':'ok'));
  document.getElementById('btxt').textContent=d.gr?'PERIGO GRAVE: Concentração crítica confirmada!':(d.at?'ATENÇÃO: Alteração detectada no laboratório':'AMBIENTE SEGURO — Condições normais');

  pill('s-combo',ensOn||ahtOn);pill('s-m2',d.m2[0]);pill('s-m135',d.m135[0]);pill('s-m7',d.m7[0]);

  const sig=SIG_INFO[d.sig]||SIG_INFO[0];
  const cls=d.gr?'gr':(d.at?'at':'');
  const detalhes=[];
  if(ensOn&&!d.ens[1]&&(d.ens[5]>=15||d.ens[6]>=15)) detalhes.push(`ENS160: TVOC ${d.ens[3]} ppb (${d.ens[5]}%) / eCO₂ ${d.ens[4]} ppm (${d.ens[6]}%)`);
  if(d.m2[0]&&d.m2[2]>=5) detalhes.push(`MQ-2: ${d.m2[1]} ppm (${d.m2[2]}%)`);
  if(d.m135[0]&&d.m135[2]>=5) detalhes.push(`MQ-135: ${d.m135[1]} ppm (${d.m135[2]}%)`);
  if(d.m7[0]&&d.m7[2]>=5) detalhes.push(`MQ-7: ${d.m7[1]} ppm (${d.m7[2]}%)`);

  const confBadge=d.sig>0?`<span class="conf-tag">Rede Neural MLP: ${d.conf}%</span>`:'';
  document.getElementById('als').innerHTML=`
    <div class="sig ${cls}">
      <strong>${sig[0]}</strong>${confBadge}<br>
      <span>${sig[1]}</span>
      ${detalhes.length?'<div class="sub-list"><b>Entradas ativas na rede:</b> '+detalhes.join(' • ')+'</div>':''}
    </div>`;

  const aqiPct=(ensOn&&!d.ens[1]&&aqiVal>0)?Math.min(100,Math.max(0,(aqiVal-1)*25)):0;
  document.getElementById('tb').innerHTML=
   row('ENS160 + AHT2X','Qualidade Geral do Ar (AQI)',AQI_TXT[aqiVal]||'Estabilizando (0%)',aqiPct,ensOn,0)+
   row('ENS160 + AHT2X','TVOC (Compostos Orgânicos / Álcool)',d.ens[3]+' ppb',d.ens[5],ensOn,0)+
   row('ENS160 + AHT2X','eCO₂ (CO₂ Equivalente)',d.ens[4]+' ppm',d.ens[6],ensOn,0)+
   row('ENS160 + AHT2X','Umidade Relativa',d.aht[2]+' %',d.aht[2],ahtOn,1)+
   row('ENS160 + AHT2X','Temperatura',d.aht[1]+' °C',d.aht[3],ahtOn,1)+
   row('MQ-135','Poluentes Gerais (NH₃, Solventes, Álcool)',d.m135[1]+' ppm (est.)',d.m135[2],d.m135[0],0)+
   row('MQ-7','Monóxido de Carbono (CO / Redutores)',d.m7[1]+' ppm (est.)',d.m7[2],d.m7[0],0)+
   row('MQ-2','Gases Inflamáveis (GLP, Álcool, Fumaça)',d.m2[1]+' ppm (est.)',d.m2[2],d.m2[0],0);

  H.co.push(d.m7[0]?d.m7[2]:0);H.nh.push(d.m135[0]?d.m135[2]:0);H.lp.push(d.m2[0]?d.m2[2]:0);
  H.vo.push(ensOn?Math.max(d.ens[5],d.ens[6]):0);H.hu.push(ahtOn?d.aht[2]:0);
  Object.values(H).forEach(a=>{if(a.length>30)a.shift();});draw();
 }catch(e){}
}
up();setInterval(up,2000);
</script></body></html>
)rawliteral";

// ===================== PERSISTÊNCIA EM MEMÓRIA NÃO VOLÁTIL (NVS) =====================
void carregarPesosNVS() {
  memcpy(W1, W1_INIT, sizeof(W1));
  memcpy(bias1, BIAS1_INIT, sizeof(bias1));
  memcpy(W2, W2_INIT, sizeof(W2));
  memcpy(bias2, BIAS2_INIT, sizeof(bias2));
  iaPersonalizada = false;

  if (prefs.getBool("has_w", false)) {
    if (prefs.getBytesLength("w1") == sizeof(W1) &&
        prefs.getBytesLength("b1") == sizeof(bias1) &&
        prefs.getBytesLength("w2") == sizeof(W2) &&
        prefs.getBytesLength("b2") == sizeof(bias2)) {
      prefs.getBytes("w1", W1, sizeof(W1));
      prefs.getBytes("b1", bias1, sizeof(bias1));
      prefs.getBytes("w2", W2, sizeof(W2));
      prefs.getBytes("b2", bias2, sizeof(bias2));
      iaPersonalizada = true;
    }
  }
}

void salvarPesosNVS() {
  prefs.putBytes("w1", W1, sizeof(W1));
  prefs.putBytes("b1", bias1, sizeof(bias1));
  prefs.putBytes("w2", W2, sizeof(W2));
  prefs.putBytes("b2", bias2, sizeof(bias2));
  prefs.putBool("has_w", true);
  iaPersonalizada = true;
}

void carregarRelatoriosNVS() {
  qtdRelatorios = prefs.getUChar("rel_cnt", 0);
  seqRelatorio = prefs.getUInt("rel_seq", 0);
  if (qtdRelatorios > MAX_REL) qtdRelatorios = 0;
  if (qtdRelatorios > 0 && prefs.getBytesLength("rels") == sizeof(relatorios)) {
    prefs.getBytes("rels", relatorios, sizeof(relatorios));
  } else {
    qtdRelatorios = 0;
  }
}

void salvarRelatoriosNVS() {
  prefs.putUChar("rel_cnt", qtdRelatorios);
  prefs.putUInt("rel_seq", seqRelatorio);
  prefs.putBytes("rels", relatorios, sizeof(relatorios));
}

// ===================== FORWARD PASS & BACKPROPAGATION DA MLP =====================
int forwardMLP(const float x[5], float probsOut[7], float hiddenOut[8]) {
  for (int i = 0; i < 8; i++) {
    float z = bias1[i];
    for (int j = 0; j < 5; j++) z += W1[i][j] * x[j];
    hiddenOut[i] = (z > 0.0f) ? z : 0.0f;
  }
  float logits[7], maxLogit = -1e9f;
  for (int k = 0; k < 7; k++) {
    float z = bias2[k];
    for (int i = 0; i < 8; i++) z += W2[k][i] * hiddenOut[i];
    logits[k] = z;
    if (z > maxLogit) maxLogit = z;
  }
  float soma = 0.0f;
  int best = 0;
  for (int k = 0; k < 7; k++) {
    probsOut[k] = expf(logits[k] - maxLogit);
    soma += probsOut[k];
    if (probsOut[k] > probsOut[best]) best = k;
  }
  if (soma > 0.0f) {
    for (int k = 0; k < 7; k++) probsOut[k] /= soma;
  }
  return best;
}

void passoBackprop(const float x[5], int targetIdx, float lr) {
  float probs[7], h[8];
  forwardMLP(x, probs, h);

  float dz2[7];
  for (int k = 0; k < 7; k++) {
    dz2[k] = probs[k] - (k == targetIdx ? 1.0f : 0.0f);
  }

  float dh[8] = {0};
  for (int k = 0; k < 7; k++) {
    for (int i = 0; i < 8; i++) {
      dh[i] += W2[k][i] * dz2[k];
      W2[k][i] -= lr * dz2[k] * h[i];
    }
    bias2[k] -= lr * dz2[k];
  }

  for (int i = 0; i < 8; i++) {
    if (h[i] <= 0.0f) continue;
    for (int j = 0; j < 5; j++) {
      W1[i][j] -= lr * dh[i] * x[j];
    }
    bias1[i] -= lr * dh[i];
  }
}

// Treina com a amostra nova + replay das outras classes para manter equilíbrio
void treinarRedeNeural(int classeAlvo, const float xNovo[5]) {
  int targetIdx = constrain(classeAlvo - 1, 0, 6);

  float replayX[7][5];
  for (int c = 0; c < 7; c++) {
    for (int j = 0; j < 5; j++) replayX[c][j] = ANCORAS[c][j];
  }
  for (int r = 0; r < qtdRelatorios; r++) {
    int c = relatorios[r].classeId - 1;
    if (c >= 0 && c < 7) {
      float m = max((float)max(relatorios[r].pTvoc, relatorios[r].pEco2),
                    (float)max(max(relatorios[r].pMq2, relatorios[r].pMq135), relatorios[r].pMq7));
      if (m > 0.0f) {
        replayX[c][0] = relatorios[r].pTvoc / m;
        replayX[c][1] = relatorios[r].pEco2 / m;
        replayX[c][2] = relatorios[r].pMq2 / m;
        replayX[c][3] = relatorios[r].pMq135 / m;
        replayX[c][4] = relatorios[r].pMq7 / m;
      }
    }
  }

  for (int ep = 0; ep < 35; ep++) {
    passoBackprop(xNovo, targetIdx, 0.12f);
    for (int c = 0; c < 7; c++) {
      if (c != targetIdx) passoBackprop(replayX[c], c, 0.04f);
    }
  }
  salvarPesosNVS();
}

// Reconstrói a rede neural do zero usando apenas os relatórios válidos restantes
void reconstruirRedeNeural() {
  memcpy(W1, W1_INIT, sizeof(W1));
  memcpy(bias1, BIAS1_INIT, sizeof(bias1));
  memcpy(W2, W2_INIT, sizeof(W2));
  memcpy(bias2, BIAS2_INIT, sizeof(bias2));

  if (qtdRelatorios == 0) {
    prefs.putBool("has_w", false);
    iaPersonalizada = false;
    return;
  }

  for (int r = 0; r < qtdRelatorios; r++) {
    int cls = relatorios[r].classeId;
    float m = max((float)max(relatorios[r].pTvoc, relatorios[r].pEco2),
                  (float)max(max(relatorios[r].pMq2, relatorios[r].pMq135), relatorios[r].pMq7));
    if (m < 1.0f) m = 1.0f;
    float x[5] = {
      relatorios[r].pTvoc / m,
      relatorios[r].pEco2 / m,
      relatorios[r].pMq2 / m,
      relatorios[r].pMq135 / m,
      relatorios[r].pMq7 / m
    };
    treinarRedeNeural(cls, x);
  }
}

void inferenciaRedeNeuralMLP(int maxP) {
  if (maxP < 36) {
    assinaturaId = 0;
    confiancaMLP = 100;
    return;
  }
  float x[5] = {
    (ensOk && !ensWarm) ? (pctTvoc / (float)maxP) : 0.0f,
    (ensOk && !ensWarm) ? (pctEco2 / (float)maxP) : 0.0f,
    mq2Ok   ? (pct2   / (float)maxP) : 0.0f,
    mq135Ok ? (pct135 / (float)maxP) : 0.0f,
    mq7Ok   ? (pct7   / (float)maxP) : 0.0f
  };
  float probs[7], h[8];
  int best = forwardMLP(x, probs, h);
  assinaturaId = best + 1;
  confiancaMLP = constrain((int)(probs[best] * 100.0f), 40, 99);
}

// ===================== LEITURA DOS SENSORES =====================
bool pingI2C(uint8_t addr) {
  Wire.beginTransmission(addr);
  return (Wire.endTransmission() == 0);
}

bool lerSensorMQ(int pino, int &mediaOut) {
  int v[25];
  for (int i = 0; i < 25; i++) {
    v[i] = analogRead(pino);
    delayMicroseconds(400);
  }
  for (int i = 0; i < 24; i++) {
    for (int j = i + 1; j < 25; j++) {
      if (v[j] < v[i]) { int t = v[i]; v[i] = v[j]; v[j] = t; }
    }
  }
  long soma = 0;
  for (int i = 2; i < 23; i++) soma += v[i];
  mediaOut = soma / 21;
  int oscilacao = v[22] - v[2];
  if (mediaOut < 10 || mediaOut > 4090 || oscilacao > 350) return false;
  return true;
}

int calcPct(float val, float base, bool ok) {
  if (!ok || base <= 0 || val - base <= 45.0f) return 0;
  float span = max(450.0f, (4095.0f - base) * 0.55f);
  return constrain((int)(((val - base) * 100.0f) / span), 1, 100);
}

int calcPPM(int pct, int seg, int grv, int maxP) {
  if (pct <= 0) return 0;
  if (pct <= 35) return map(pct, 1, 35, 1, seg);
  if (pct <= 69) return map(pct, 36, 69, seg + 1, grv);
  return map(pct, 70, 100, grv + 1, maxP);
}

void lerSensores() {
  bool ahtAgora = pingI2C(0x38);
  if (ahtAgora && !ahtOk) aht.begin();
  ahtOk = ahtAgora;

  if (ahtOk) {
    sensors_event_t h, t;
    if (aht.getEvent(&h, &t)) {
      tempC = t.temperature;
      umidPct = constrain(h.relative_humidity, 0.0f, 100.0f);
      pctTemp = constrain((int)((tempC - 10.0f) * 2.5f), 0, 100);
    } else ahtOk = false;
  }

  bool ensAgora = pingI2C(0x53);
  if (ensAgora && !ensOk) {
    if (ens160.begin()) ens160.setMode(ENS160_OPMODE_STD);
  }
  ensOk = ensAgora;

  if (ensOk) {
    if (ahtOk) ens160.set_envdata(tempC, umidPct);
    ens160.measure(true);
    uint8_t rawAqi = ens160.getAQI();
    uint16_t rawTvoc = ens160.getTVOC();
    uint16_t rawEco2 = ens160.geteCO2();

    if (rawAqi == 0 || rawAqi == 255 || rawTvoc == 65535 || rawEco2 == 65535) {
      ensWarm = true; aqi = 0; tvoc = 0; eco2 = 400; pctTvoc = 0; pctEco2 = 0;
    } else {
      ensWarm = false; aqi = rawAqi; tvoc = rawTvoc; eco2 = rawEco2;
      pctTvoc = constrain(map(tvoc, 0, 1500, 0, 100), 0, 100);
      pctEco2 = constrain(map(eco2, 400, 2500, 0, 100), 0, 100);
      if (aqi >= 5) pctTvoc = max(pctTvoc, 75);
    }
  } else {
    ensWarm = false; aqi = 0; tvoc = 0; eco2 = 400; pctTvoc = 0; pctEco2 = 0;
  }

  int r2 = 0, r135 = 0, r7 = 0;
  mq2Ok   = lerSensorMQ(PIN_MQ2, r2);
  mq135Ok = lerSensorMQ(PIN_MQ135, r135);
  mq7Ok   = lerSensorMQ(PIN_MQ7, r7);

  if (mq2Ok) {
    if (base2 == 0) { base2 = r2; filt2 = r2; }
    filt2 = 0.65f * filt2 + 0.35f * r2;
    if (fabsf(filt2 - base2) < 35.0f) base2 = 0.98f * base2 + 0.02f * filt2;
  } else { base2 = 0; filt2 = 0; }

  if (mq135Ok) {
    if (base135 == 0) { base135 = r135; filt135 = r135; }
    filt135 = 0.65f * filt135 + 0.35f * r135;
    if (fabsf(filt135 - base135) < 35.0f) base135 = 0.98f * base135 + 0.02f * filt135;
  } else { base135 = 0; filt135 = 0; }

  if (mq7Ok) {
    if (base7 == 0) { base7 = r7; filt7 = r7; }
    filt7 = 0.65f * filt7 + 0.35f * r7;
    if (fabsf(filt7 - base7) < 35.0f) base7 = 0.98f * base7 + 0.02f * filt7;
  } else { base7 = 0; filt7 = 0; }

  pct2   = calcPct(filt2, base2, mq2Ok);
  pct135 = calcPct(filt135, base135, mq135Ok);
  pct7   = calcPct(filt7, base7, mq7Ok);

  ppm2   = mq2Ok   ? calcPPM(pct2, 300, 1500, 10000) : 0;
  ppm135 = mq135Ok ? calcPPM(pct135, 25, 80, 300)    : 0;
  ppm7   = mq7Ok   ? calcPPM(pct7, 35, 100, 2000)    : 0;

  int maxP = max(max(pct2, pct135), pct7);
  if (ensOk && !ensWarm) maxP = max(maxP, max(pctTvoc, pctEco2));

  if (maxP >= 70) contGrave++; else if (maxP < 64) contGrave = 0;
  if (maxP >= 36) contAtencao++; else if (maxP < 32) contAtencao = 0;

  grave   = (contGrave >= 2);
  atencao = (!grave && contAtencao >= 2);
  inferenciaRedeNeuralMLP((atencao || grave) ? maxP : 0);

  digitalWrite(PIN_LED_R, (atencao || grave) ? HIGH : LOW);
  digitalWrite(PIN_LED_G, (atencao || grave) ? LOW : HIGH);
}

void tocarBuzzers() {
  bool emTeste = (tTesteBuz > 0 && millis() < tTesteBuz);
  if ((!grave || modoTreinoAtivo) && !emTeste) {
    analogWrite(PIN_BUZ1, 0);
    analogWrite(PIN_BUZ2, 0);
    return;
  }
  int pwm = (volBuzzerPct <= 0) ? 0 : map(volBuzzerPct, 1, 100, 15, 255);
  if (millis() - tBuz >= 280) {
    tBuz = millis();
    passoBuz = (passoBuz + 1) % 4;
    analogWrite(PIN_BUZ1, passoBuz == 0 ? pwm : 0);
    analogWrite(PIN_BUZ2, passoBuz == 1 ? pwm : 0);
  }
}

// ===================== ROTAS WEB =====================
void handleRoot() { server.send_P(200, "text/html", HTML); }

void handleModo() {
  modoTreinoAtivo = (server.arg("m").toInt() == 1);
  server.send(200, "text/plain", "OK");
}

void handleVolume() {
  if (server.hasArg("v")) {
    volBuzzerPct = constrain(server.arg("v").toInt(), 0, 100);
    prefs.putInt("buz_vol", volBuzzerPct);
  }
  if (server.arg("beep").toInt() == 1 && volBuzzerPct > 0) {
    tTesteBuz = millis() + 650;
  }
  server.send(200, "text/plain", String(volBuzzerPct));
}

void handleDados() {
  char buf[580];
  snprintf(buf, sizeof(buf),
    "{\"ip\":\"%s\",\"iap\":%d,\"vol\":%d,\"at\":%d,\"gr\":%d,\"sig\":%d,\"conf\":%d,\"ens\":[%d,%d,%u,%u,%u,%d,%d],\"aht\":[%d,%.1f,%.1f,%d],\"m2\":[%d,%d,%d],\"m135\":[%d,%d,%d],\"m7\":[%d,%d,%d]}",
    WiFi.localIP().toString().c_str(), iaPersonalizada, volBuzzerPct, atencao, grave, assinaturaId, confiancaMLP,
    ensOk, ensWarm, aqi, tvoc, eco2, pctTvoc, pctEco2,
    ahtOk, tempC, umidPct, pctTemp,
    mq2Ok, ppm2, pct2,
    mq135Ok, ppm135, pct135,
    mq7Ok, ppm7, pct7);
  server.send(200, "application/json", buf);
}

void handleTreinar() {
  int cls = constrain(server.arg("cls").toInt(), 1, 7);
  int pt  = constrain(server.arg("pt").toInt(), 0, 100);
  int pe  = constrain(server.arg("pe").toInt(), 0, 100);
  int p2  = constrain(server.arg("p2").toInt(), 0, 100);
  int p135= constrain(server.arg("p135").toInt(), 0, 100);
  int p7  = constrain(server.arg("p7").toInt(), 0, 100);

  float maxP = max((float)max(pt, pe), (float)max(max(p2, p135), p7));
  if (maxP < 10.0f) {
    server.send(200, "application/json",
      "{\"erro\":\"Nenhum sensor detectou alteração significativa (mínimo 10%). Aproxime mais a substância e tente novamente.\"}");
    return;
  }

  float xNovo[5] = { pt / maxP, pe / maxP, p2 / maxP, p135 / maxP, p7 / maxP };
  float probs[7], h[8];

  forwardMLP(xNovo, probs, h);
  int confAntes = constrain((int)(probs[cls - 1] * 100.0f), 1, 99);

  treinarRedeNeural(cls, xNovo);

  forwardMLP(xNovo, probs, h);
  int confDepois = constrain((int)(probs[cls - 1] * 100.0f), 1, 99);

  if (qtdRelatorios >= MAX_REL) {
    for (int i = 0; i < MAX_REL - 1; i++) relatorios[i] = relatorios[i + 1];
    qtdRelatorios = MAX_REL - 1;
  }
  seqRelatorio++;
  RelatorioTreino &r = relatorios[qtdRelatorios++];
  r.id = seqRelatorio;
  r.tempoSeg = millis() / 1000;
  r.classeId = cls;
  r.confAntes = confAntes;
  r.confDepois = confDepois;
  r.pTvoc = pt; r.pEco2 = pe; r.pMq2 = p2; r.pMq135 = p135; r.pMq7 = p7;
  r.rawTvoc = server.arg("rt").toInt();
  r.rawEco2 = server.arg("re").toInt();
  r.ppm2 = server.arg("ppm2").toInt();
  r.ppm135 = server.arg("ppm135").toInt();
  r.ppm7 = server.arg("ppm7").toInt();
  String nt = server.arg("nota");
  nt.replace("\"", "'"); nt.replace(";", ",");
  nt.toCharArray(r.nota, sizeof(r.nota));

  salvarRelatoriosNVS();

  char resp[220];
  snprintf(resp, sizeof(resp), "{\"id\":%u,\"nome\":\"%s\",\"antes\":%d,\"depois\":%d}",
           r.id, NOMES_CLASSES[cls], confAntes, confDepois);
  server.send(200, "application/json", resp);
}

void handleExcluirRel() {
  uint32_t idAlvo = server.arg("id").toInt();
  int idx = -1;
  for (int i = 0; i < qtdRelatorios; i++) {
    if (relatorios[i].id == idAlvo) {
      idx = i;
      break;
    }
  }
  if (idx >= 0) {
    for (int i = idx; i < qtdRelatorios - 1; i++) {
      relatorios[i] = relatorios[i + 1];
    }
    qtdRelatorios--;
    salvarRelatoriosNVS();
    reconstruirRedeNeural();
  }
  server.send(200, "text/plain", "OK");
}

void handleRelatorios() {
  String out;
  out.reserve(2400);
  out = "[";
  for (int i = 0; i < qtdRelatorios; i++) {
    RelatorioTreino &r = relatorios[i];
    if (i > 0) out += ",";
    char item[240];
    snprintf(item, sizeof(item),
      "{\"id\":%u,\"t\":%u,\"nome\":\"%s\",\"nota\":\"%s\",\"pt\":%u,\"pe\":%u,\"p2\":%u,\"p135\":%u,\"p7\":%u,\"ca\":%u,\"cd\":%u}",
      r.id, r.tempoSeg, NOMES_CLASSES[r.classeId], r.nota,
      r.pTvoc, r.pEco2, r.pMq2, r.pMq135, r.pMq7,
      r.confAntes, r.confDepois);
    out += item;
  }
  out += "]";
  server.send(200, "application/json", out);
}

void handleCSV() {
  String csv;
  csv.reserve(2600);
  csv = "ID;Tempo_s;Substancia;Observacao;TVOC_ppb;TVOC_pct;eCO2_ppm;eCO2_pct;MQ2_ppm;MQ2_pct;MQ135_ppm;MQ135_pct;MQ7_ppm;MQ7_pct;Conf_Antes_pct;Conf_Depois_pct\n";
  for (int i = 0; i < qtdRelatorios; i++) {
    RelatorioTreino &r = relatorios[i];
    char line[200];
    snprintf(line, sizeof(line), "%u;%u;%s;%s;%u;%u;%u;%u;%d;%u;%d;%u;%d;%u;%u;%u\n",
      r.id, r.tempoSeg, NOMES_CLASSES[r.classeId], r.nota,
      r.rawTvoc, r.pTvoc, r.rawEco2, r.pEco2,
      r.ppm2, r.pMq2, r.ppm135, r.pMq135, r.ppm7, r.pMq7,
      r.confAntes, r.confDepois);
    csv += line;
  }
  server.send(200, "text/csv; charset=utf-8", csv);
}

void handleResetIA() {
  prefs.putBool("has_w", false);
  carregarPesosNVS();
  server.send(200, "text/plain", "OK");
}

void handleResetRel() {
  qtdRelatorios = 0;
  seqRelatorio = 0;
  salvarRelatoriosNVS();
  reconstruirRedeNeural();
  server.send(200, "text/plain", "OK");
}

void setup() {
  Serial.begin(9600);
  prefs.begin("lab_ia", false);
  volBuzzerPct = constrain(prefs.getInt("buz_vol", 15), 0, 100);
  carregarPesosNVS();
  carregarRelatoriosNVS();

  pinMode(PIN_LED_G, OUTPUT); pinMode(PIN_LED_R, OUTPUT);
  pinMode(PIN_BUZ1, OUTPUT);  pinMode(PIN_BUZ2, OUTPUT);
  digitalWrite(PIN_LED_G, HIGH); digitalWrite(PIN_LED_R, LOW);
  analogWrite(PIN_BUZ1, 0); analogWrite(PIN_BUZ2, 0);

  pinMode(PIN_MQ2, INPUT);
  pinMode(PIN_MQ135, INPUT);
  pinMode(PIN_MQ7, INPUT);

  Wire.begin(21, 22);
  ahtOk = pingI2C(0x38) && aht.begin();
  ensOk = pingI2C(0x53) && ens160.begin();
  if (ensOk) ens160.setMode(ENS160_OPMODE_STD);

  Serial.println("Calibrando sensores MQ ativos (3s)...");
  long s2 = 0, s135 = 0, s7 = 0;
  int c2 = 0, c135 = 0, c7 = 0;
  for (int i = 0; i < 15; i++) {
    int v;
    if (lerSensorMQ(PIN_MQ2, v))   { s2 += v; c2++; }
    if (lerSensorMQ(PIN_MQ135, v)) { s135 += v; c135++; }
    if (lerSensorMQ(PIN_MQ7, v))   { s7 += v; c7++; }
    delay(150);
  }
  base2   = (c2 > 7)   ? (s2 / c2)     : 0; filt2   = base2;
  base135 = (c135 > 7) ? (s135 / c135) : 0; filt135 = base135;
  base7   = (c7 > 7)   ? (s7 / c7)     : 0; filt7   = base7;

  WiFi.mode(WIFI_STA);
  WiFi.begin(SSID, PASS);
  Serial.print("Conectando Wi-Fi");
  for (int i = 0; i < 30 && WiFi.status() != WL_CONNECTED; i++) { delay(500); Serial.print("."); }
  Serial.print("\nIP: http://"); Serial.println(WiFi.localIP());

  server.on("/", handleRoot);
  server.on("/modo", handleModo);
  server.on("/volume", handleVolume);
  server.on("/dados", handleDados);
  server.on("/treinar", handleTreinar);
  server.on("/excluir_rel", handleExcluirRel);
  server.on("/relatorios", handleRelatorios);
  server.on("/relatorios.csv", handleCSV);
  server.on("/reset_ia", handleResetIA);
  server.on("/reset_rel", handleResetRel);
  server.begin();
  lerSensores();
}

void loop() {
  server.handleClient();
  tocarBuzzers();
  unsigned long now = millis();
  if (now - tWifi >= 10000) {
    tWifi = now;
    if (WiFi.status() != WL_CONNECTED) WiFi.reconnect();
  }
  if (now - tLeitura >= 2000) {
    tLeitura = now;
    lerSensores();
    Serial.printf("MLP:%d(%d%%) | CO:%d%% | NH3:%d%% | GLP:%d%% | TVOC:%uppb | Vol:%d%%\n",
                  assinaturaId, confiancaMLP, pct7, pct135, pct2, tvoc, volBuzzerPct);
  }
}
