"""Página de demostración: se sube una foto y se ven las detecciones, la especie, la confianza y el tiempo."""

PAGINA_DEMO = """<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IdentiMar — demostración</title>
<style>body{font-family:system-ui,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#10263a;background:#f6f2ea}
h1{font-size:26px}label{display:block;margin:10px 0 4px;font-weight:600}input{padding:8px;width:100%;box-sizing:border-box}
button{margin-top:14px;padding:10px 20px;background:#1f8a8f;color:#fff;border:0;border-radius:8px;font-size:16px;cursor:pointer}
#r{display:flex;gap:24px;flex-wrap:wrap;margin-top:20px}#r img{max-width:520px;width:100%;border-radius:10px;background:#fff}
#c{background:#fff;padding:16px;border-radius:10px;min-width:260px;flex:1}.ok{color:#1b7f3a}.dud{color:#b45a1f}pre{white-space:pre-wrap;font-size:13px}</style></head>
<body><h1>IdentiMar — identificación de rayas</h1>
<label for="k">API key</label><input id="k" type="password" autocomplete="off">
<label for="f">Fotografía (JPEG, PNG o WEBP)</label><input id="f" type="file" accept="image/*">
<button id="b">Identificar</button><div id="m"></div>
<div id="r" hidden><img id="i" alt="Detecciones"><div id="c"></div></div>
<script>
const $=id=>document.getElementById(id);
$("b").onclick=async()=>{
  const file=$("f").files[0]; if(!file){$("m").textContent="Elija una fotografía.";return}
  const fd=new FormData(); fd.append("id_observacion","demo-"+Date.now()); fd.append("imagen",file);
  $("m").textContent="Procesando…"; const t0=performance.now();
  const r=await fetch("/predict/visual",{method:"POST",headers:$("k").value?{"X-API-Key":$("k").value}:{},body:fd});
  if(!r.ok){$("m").textContent="Error "+r.status+": "+(await r.text());return}
  const j=JSON.parse(r.headers.get("X-Resultado")); $("i").src=URL.createObjectURL(await r.blob());
  const ms=Math.round(performance.now()-t0), cls=j.estado==="identificado"?"ok":"dud";
  $("c").innerHTML="<h2 class='"+cls+"'>"+j.estado.replace("_"," ").toUpperCase()+"</h2><p><b>Especie propuesta:</b> "+(j.especie_propuesta||"ninguna")+
   "</p><p><b>Confianza:</b> "+(j.confianza*100).toFixed(1)+" % (umbral "+(j.umbral_minimo*100).toFixed(0)+" %)</p><p><b>Inferencia:</b> "+j.tiempo_inferencia_ms+
   " ms · <b>Total (con red):</b> "+ms+" ms</p><p><b>Modelo:</b> "+j.version_modelo+"</p><pre>"+JSON.stringify(j,null,2)+"</pre>";
  $("r").hidden=false; $("m").textContent="";
};
</script></body></html>"""
