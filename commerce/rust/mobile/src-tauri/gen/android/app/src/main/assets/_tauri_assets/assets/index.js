(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const s of document.querySelectorAll('link[rel="modulepreload"]'))o(s);new MutationObserver(s=>{for(const i of s)if(i.type==="childList")for(const n of i.addedNodes)n.tagName==="LINK"&&n.rel==="modulepreload"&&o(n)}).observe(document,{childList:!0,subtree:!0});function r(s){const i={};return s.integrity&&(i.integrity=s.integrity),s.referrerPolicy&&(i.referrerPolicy=s.referrerPolicy),s.crossOrigin==="use-credentials"?i.credentials="include":s.crossOrigin==="anonymous"?i.credentials="omit":i.credentials="same-origin",i}function o(s){if(s.ep)return;s.ep=!0;const i=r(s);fetch(s.href,i)}})();function V(e){const t=new Date(e),o=Math.floor((new Date().getTime()-t.getTime())/1e3),s=o/31536e3;if(s>1)return Math.floor(s)+" years";const i=o/2592e3;if(i>1)return Math.floor(i)+" months";const n=o/86400;if(n>1)return Math.floor(n)+" days";const c=o/3600;if(c>1)return Math.floor(c)+" hours";const y=o/60;return y>1?Math.floor(y)+" minutes":Math.floor(o)+" seconds"}function j(e){if(typeof e=="string"){if(isNaN(Number(e)))return e;e=Number(e)}let t="";switch(e){case 1:t="progress";break;case 2:t="stop";break;case 3:t="cancel";break;case 4:t="refund";break;case 5:t="return";break;case 6:t="error";break;case 7:t="expire";break;case 8:t="exchange";break;case 9:t="complete";break;case 10:t="draft";break;case 11:t="show";break;case 12:t="hide";break;default:t="unknown"}return t}const m={result:"logis-result",info:"logis-info",more:"logis-more",created_at:"field-created-at"};function Q(e,t,r=""){let o=!1,s="";e.data&&e.data.link?s=e.data.link:e.link&&(s=e.link),s&&r&&r.includes(s)&&(o=!0);let i=e.type||e.doc_type||"unknown";i==="sales"||i==="goods"||i==="order"?i="sales":i==="event"||i==="coupon"?i="event":(i==="receiving"||i==="shipping")&&(i="tracking");function n(l,f,b=""){let d="",_="",x=f.replace(/_/g," ");if(l[f]!==void 0?d=l[f]:l.data&&l.data[f]!==void 0&&(d=l.data[f]),d===""||d===null||d===void 0)return"";f==="status"&&(d=j(d),x=l.type||"Status"),b&&(l[b]!==void 0?_=` (${l[b]})`:l.data&&l.data[b]!==void 0&&(_=` (${l.data[b]})`)),["created_at","updated_at","started_at","expired_at","release_date"].includes(f)&&(d=V(d));let S="div",q="",T="";return f==="title"&&(S="a",(l.link||l.data&&l.data.link)&&(q=`href="#" onclick="document.dispatchEvent(new CustomEvent('nav-link', {detail: '${l.link||l.data.link}'})); return false;"`)),T=`<span class="value">${d}</span><i class="unit">${_}</i>`,`
            <${S} ${q} class="${m.info} ${f}">
                <strong>${x}</strong>
                ${T}
            </${S}>
        `}let c=`<div class="${m.result} ${i}" id="${e.uuid||e.id}">`;const y=`more-${e.id||Math.random().toString(36).substr(2,9)}`;return c+=`<input type="checkbox" id="${y}" class="toggle-more" ${o?"checked":""} style="display:none;" />`,i==="sales"?c+=`
            ${n(e,"status")}
            ${n(e,"title")}
            ${n(e,"sale_price","currency")}
            ${n(e,"created_at")}
            <label for="${y}" class="more-label">▼ details</label>
            <div class="${m.more}">
                ${n(e,"price","currency")}
                ${n(e,"quantity")}
                ${n(e,"stock_keeping_unit")}
                ${n(e,"shipping_fee","currency")}
                ${n(e,"shipping_method")}
                ${n(e,"tax_included")}
            </div>
        `:i==="tracking"?c+=`
            ${n(e,"status")}
            ${n(e,"title")}
            ${n(e,"carrier")}
            ${n(e,"created_at")}
            <label for="${y}" class="more-label">▼ details</label>
            <div class="${m.more}">
                ${n(e,"text")}
                ${n(e,"sender_name")}
                ${n(e,"recipient_name")}
                ${n(e,"tracking_number")}
            </div>
        `:i==="event"?c+=`
            ${n(e,"status")}
            ${n(e,"title")}
            ${n(e,"discount")}
            ${n(e,"expired_at")}
            <label for="${y}" class="more-label">▼ details</label>
            <div class="${m.more}">
                ${n(e,"code")}
                ${n(e,"min_order_amount")}
                ${n(e,"usage_limit")}
            </div>
        `:c+=`
            ${n(e,"id")}
            ${n(e,"type")}
            ${n(e,"created_at")}
            <div style="font-size:0.8rem; padding:10px; color:#666;">${e.text||""}</div>
        `,c+=`<input type="hidden" readonly name="${m.created_at}" value="${e.created_at}" />`,c+="</div>",c}function $(e){console.log(`[Main] ${e}`);const t=document.getElementById("log-panel");if(t){const r=document.createElement("div");r.textContent=`> ${e}`,t.appendChild(r),t.scrollTop=t.scrollHeight}}$("Main module loaded. V137 (Inside Panel Intro) Initializing...");let E=null,k=!1,a=null,u=null,v=[],I=0,p=null;const h=document.getElementById("v"),g=document.getElementById("global-search"),U=document.querySelectorAll(".tab-content"),M=document.getElementById("list-view"),O=document.getElementById("detail-view"),w=document.querySelector(".chat-form"),z=document.querySelector(".chat-talks"),B=document.getElementById("chat-scroll");function L(e){$(`Tab -> ${e}`),U.forEach(t=>{t.id===`tab-${e}`?(t.classList.add("active"),t.style.display="flex"):(t.classList.remove("active"),t.style.display="none")})}function H(e){e?(M.style.display="none",O.style.display="flex"):(M.style.display="flex",O.style.display="none")}async function F(){if(!k){v=[],I=0,p&&(clearInterval(p),p=null);try{const e={video:{facingMode:"environment"}};E=await navigator.mediaDevices.getUserMedia(e),h.srcObject=E,await h.play(),k=!0,document.getElementById("scanner-overlay").style.display="block",requestAnimationFrame(J)}catch(e){$(`Camera Err: ${e}`)}}}function P(){k=!1,E&&E.getTracks().forEach(e=>e.stop()),document.getElementById("scanner-overlay").style.display="none"}function J(){if(k){if(h.readyState===h.HAVE_ENOUGH_DATA){const e=document.createElement("canvas");e.width=h.videoWidth,e.height=h.videoHeight;const t=e.getContext("2d");if(t){t.drawImage(h,0,0,e.width,e.height);const r=t.getImageData(0,0,e.width,e.height),o=jsQR(r.data,r.width,r.height);o&&K(o.data)}}requestAnimationFrame(J)}}function K(e){try{const t=JSON.parse(e);if(Array.isArray(t)&&t.length===3){const[r,o,s]=t;if(I===0&&(I=o,v=new Array(o).fill("")),!v[r]){v[r]=s,$(`Offer ${r+1}/${o}`);const i=document.querySelector("#tab-intro p");i&&(i.textContent=`Scanning... ${v.filter(n=>n).length}/${o}`)}v.every(i=>i!=="")&&(P(),G(v.join("")))}}catch{}}async function G(e){u=new RTCPeerConnection({iceServers:[]}),u.ondatachannel=n=>{a=n.channel,Y(a)},u.oniceconnectionstatechange=()=>{(u==null?void 0:u.iceConnectionState)==="connected"&&($("🚀 Connected! Switching to List UI..."),p&&(clearInterval(p),p=null),document.getElementById("answer-qr-container").style.display="none",L("tab-list"),g&&(g.disabled=!1,g.placeholder="Search or Ask Prompt"),document.querySelectorAll(".nav-icons .nav-btn").forEach(n=>n.disabled=!1))},await u.setRemoteDescription(new RTCSessionDescription({type:"offer",sdp:e}));const t=await u.createAnswer();await u.setLocalDescription(t);const r=u.localDescription.sdp,o=4,s=Math.ceil(r.length/o),i=[];for(let n=0;n<o;n++)i.push(JSON.stringify([n,o,r.substring(n*s,(n+1)*s)]));W(i)}function W(e){const t=document.getElementById("answer-qr-container");t.style.display="flex";const r=document.getElementById("answer-qr");let o=0;const s=()=>{r.innerHTML=`<div style="font-weight:bold; margin-bottom:10px;">Part ${o+1}/${e.length}</div>`;const i=document.createElement("div");r.appendChild(i),new QRCode(i,{text:e[o],width:250,height:250}),o=(o+1)%e.length};p&&clearInterval(p),s(),p=setInterval(s,1e3)}function Y(e){e.onopen=()=>{$("Linked!"),e.send(JSON.stringify({type:"search",query:""}))},e.onmessage=t=>{const r=JSON.parse(t.data);r.type==="sync_list"?X(r.data):r.type==="sync_detail"?Z(r.title,r.content):r.type==="sync_chat"&&R(r.data)}}function X(e){const t=document.getElementById("doc-list");t&&(t.innerHTML=e.map(r=>Q(r)).join(""),t.querySelectorAll(".logis-result").forEach(r=>{r.addEventListener("click",()=>{a==null||a.send(JSON.stringify({type:"get_detail",uuid:r.id}))})}))}function Z(e,t){document.getElementById("detail-title").innerText=e,document.getElementById("detail-content").innerHTML=t,H(!0)}function R(e){const t=document.createElement("div");t.className=`chat-talk ${e.role==="user"?"user":"system"}`,t.innerHTML=`<div class="chat-message"><div class="content">${e.content}</div></div>`,z.appendChild(t),B.scrollTop=B.scrollHeight}g==null||g.addEventListener("input",e=>{a==null||a.send(JSON.stringify({type:"search",query:e.target.value}))});w==null||w.addEventListener("submit",e=>{e.preventDefault();const t=w.querySelector('input[name="talk"]');t.value.trim()&&(R({role:"user",content:t.value}),a==null||a.send(JSON.stringify({type:"chat_message",content:t.value})),t.value="")});var N;(N=document.getElementById("btn-settings"))==null||N.addEventListener("click",()=>L("tab-settings"));var D;(D=document.getElementById("btn-settings-back"))==null||D.addEventListener("click",()=>L("tab-list"));var A;(A=document.getElementById("btn-detail-back"))==null||A.addEventListener("click",()=>H(!1));var C;(C=document.getElementById("list-refresh-btn"))==null||C.addEventListener("click",()=>{a==null||a.send(JSON.stringify({type:"search",query:(g==null?void 0:g.value)||""}))});window.addEventListener("request-camera-start",()=>F());window.addEventListener("request-camera-stop",()=>P());
