(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const s of document.querySelectorAll('link[rel="modulepreload"]'))o(s);new MutationObserver(s=>{for(const r of s)if(r.type==="childList")for(const n of r.addedNodes)n.tagName==="LINK"&&n.rel==="modulepreload"&&o(n)}).observe(document,{childList:!0,subtree:!0});function i(s){const r={};return s.integrity&&(r.integrity=s.integrity),s.referrerPolicy&&(r.referrerPolicy=s.referrerPolicy),s.crossOrigin==="use-credentials"?r.credentials="include":s.crossOrigin==="anonymous"?r.credentials="omit":r.credentials="same-origin",r}function o(s){if(s.ep)return;s.ep=!0;const r=i(s);fetch(s.href,r)}})();function Q(e){const t=new Date(e),o=Math.floor((new Date().getTime()-t.getTime())/1e3),s=o/31536e3;if(s>1)return Math.floor(s)+" years";const r=o/2592e3;if(r>1)return Math.floor(r)+" months";const n=o/86400;if(n>1)return Math.floor(n)+" days";const c=o/3600;if(c>1)return Math.floor(c)+" hours";const p=o/60;return p>1?Math.floor(p)+" minutes":Math.floor(o)+" seconds"}function V(e){if(typeof e=="string"){if(isNaN(Number(e)))return e;e=Number(e)}let t="";switch(e){case 1:t="progress";break;case 2:t="stop";break;case 3:t="cancel";break;case 4:t="refund";break;case 5:t="return";break;case 6:t="error";break;case 7:t="expire";break;case 8:t="exchange";break;case 9:t="complete";break;case 10:t="draft";break;case 11:t="show";break;case 12:t="hide";break;default:t="unknown"}return t}const v={result:"logis-result",info:"logis-info",more:"logis-more",created_at:"field-created-at"};function j(e,t,i=""){let o=!1,s="";e.data&&e.data.link?s=e.data.link:e.link&&(s=e.link),s&&i&&i.includes(s)&&(o=!0);let r=e.type||e.doc_type||"unknown";r==="sales"||r==="goods"||r==="order"?r="sales":r==="event"||r==="coupon"?r="event":(r==="receiving"||r==="shipping")&&(r="tracking");function n(l,f,$=""){let d="",_="",L=f.replace(/_/g," ");if(l[f]!==void 0?d=l[f]:l.data&&l.data[f]!==void 0&&(d=l.data[f]),d===""||d===null||d===void 0)return"";f==="status"&&(d=V(d),L=l.type||"Status"),$&&(l[$]!==void 0?_=` (${l[$]})`:l.data&&l.data[$]!==void 0&&(_=` (${l.data[$]})`)),["created_at","updated_at","started_at","expired_at","release_date"].includes(f)&&(d=Q(d));let S="div",x="",q="";return f==="title"&&(S="a",(l.link||l.data&&l.data.link)&&(x=`href="#" onclick="document.dispatchEvent(new CustomEvent('nav-link', {detail: '${l.link||l.data.link}'})); return false;"`)),q=`<span class="value">${d}</span><i class="unit">${_}</i>`,`
            <${S} ${x} class="${v.info} ${f}">
                <strong>${L}</strong>
                ${q}
            </${S}>
        `}let c=`<div class="${v.result} ${r}" id="${e.uuid||e.id}">`;const p=`more-${e.id||Math.random().toString(36).substr(2,9)}`;return c+=`<input type="checkbox" id="${p}" class="toggle-more" ${o?"checked":""} style="display:none;" />`,r==="sales"?c+=`
            ${n(e,"status")}
            ${n(e,"title")}
            ${n(e,"sale_price","currency")}
            ${n(e,"created_at")}
            <label for="${p}" class="more-label">▼ details</label>
            <div class="${v.more}">
                ${n(e,"price","currency")}
                ${n(e,"quantity")}
                ${n(e,"stock_keeping_unit")}
                ${n(e,"shipping_fee","currency")}
                ${n(e,"shipping_method")}
                ${n(e,"tax_included")}
            </div>
        `:r==="tracking"?c+=`
            ${n(e,"status")}
            ${n(e,"title")}
            ${n(e,"carrier")}
            ${n(e,"created_at")}
            <label for="${p}" class="more-label">▼ details</label>
            <div class="${v.more}">
                ${n(e,"text")}
                ${n(e,"sender_name")}
                ${n(e,"recipient_name")}
                ${n(e,"tracking_number")}
            </div>
        `:r==="event"?c+=`
            ${n(e,"status")}
            ${n(e,"title")}
            ${n(e,"discount")}
            ${n(e,"expired_at")}
            <label for="${p}" class="more-label">▼ details</label>
            <div class="${v.more}">
                ${n(e,"code")}
                ${n(e,"min_order_amount")}
                ${n(e,"usage_limit")}
            </div>
        `:c+=`
            ${n(e,"id")}
            ${n(e,"type")}
            ${n(e,"created_at")}
            <div style="font-size:0.8rem; padding:10px; color:#666;">${e.text||""}</div>
        `,c+=`<input type="hidden" readonly name="${v.created_at}" value="${e.created_at}" />`,c+="</div>",c}function b(e){console.log(`[Main] ${e}`);const t=document.getElementById("log-panel");if(t){const i=document.createElement("div");i.textContent=`> ${e}`,t.appendChild(i),t.scrollTop=t.scrollHeight}}b("Main module loaded. V134 (Slide QR) Initializing...");let E=null,k=!1,a=null,u=null,y=[],I=0,g=null;const m=document.getElementById("v"),h=document.getElementById("global-search"),z=document.querySelectorAll(".tab-content"),M=document.getElementById("list-view"),N=document.getElementById("detail-view"),w=document.querySelector('form[name="chat-form"]'),U=document.querySelector(".chat-talks"),O=document.getElementById("chat-scroll");function A(e){z.forEach(t=>{t.id===`tab-${e}`?t.classList.add("active"):t.classList.remove("active")})}function H(e){e?(M.style.display="none",N.style.display="flex"):(M.style.display="flex",N.style.display="none")}async function F(){if(!k){y=[],I=0,g&&(clearInterval(g),g=null);try{const e={video:{facingMode:"environment"}};E=await navigator.mediaDevices.getUserMedia(e),m.srcObject=E,await m.play(),k=!0,document.getElementById("scanner-overlay").style.display="block",requestAnimationFrame(J)}catch(e){b(`Camera Err: ${e}`)}}}function P(){k=!1,E&&E.getTracks().forEach(e=>e.stop()),document.getElementById("scanner-overlay").style.display="none"}function J(){if(k){if(m.readyState===m.HAVE_ENOUGH_DATA){const e=document.createElement("canvas");e.width=m.videoWidth,e.height=m.videoHeight;const t=e.getContext("2d");if(t){t.drawImage(m,0,0,e.width,e.height);const i=t.getImageData(0,0,e.width,e.height),o=jsQR(i.data,i.width,i.height);o&&K(o.data)}}requestAnimationFrame(J)}}function K(e){try{const t=JSON.parse(e);if(Array.isArray(t)&&t.length===3){const[i,o,s]=t;if(I===0&&(I=o,y=new Array(o).fill("")),!y[i]){y[i]=s,b(`Part ${i+1}/${o} Received`);const r=document.querySelector("#mobile-intro-overlay p");r&&(r.textContent=`Scanning... ${y.filter(n=>n).length}/${o}`)}y.every(r=>r!=="")&&(b("All Offer parts received. Connecting..."),P(),G(y.join("")))}}catch{}}async function G(e){u=new RTCPeerConnection({iceServers:[]}),u.ondatachannel=n=>{a=n.channel,Y(a)},u.oniceconnectionstatechange=()=>{(u==null?void 0:u.iceConnectionState)==="connected"&&(document.getElementById("mobile-intro-overlay").style.display="none",g&&(clearInterval(g),g=null),document.getElementById("answer-qr-container").style.display="none")},await u.setRemoteDescription(new RTCSessionDescription({type:"offer",sdp:e}));const t=await u.createAnswer();await u.setLocalDescription(t);const i=u.localDescription.sdp,o=4,s=Math.ceil(i.length/o),r=[];for(let n=0;n<o;n++)r.push(JSON.stringify([n,o,i.substring(n*s,(n+1)*s)]));W(r)}function W(e){const t=document.getElementById("answer-qr-container");t.style.display="flex";const i=document.getElementById("answer-qr");let o=0;const s=()=>{i.innerHTML=`<div style="font-weight:bold; margin-bottom:10px;">Answer Part ${o+1}/${e.length}</div>`;const r=document.createElement("div");i.appendChild(r),new QRCode(r,{text:e[o],width:250,height:250}),o=(o+1)%e.length};g&&clearInterval(g),s(),g=setInterval(s,1e3)}function Y(e){e.onopen=()=>{b("Channel OPEN - Connected!"),e.send(JSON.stringify({type:"search",query:""}))},e.onmessage=t=>{const i=JSON.parse(t.data);i.type==="sync_list"?X(i.data):i.type==="sync_detail"?Z(i.title,i.content):i.type==="sync_chat"&&R(i.data)}}function X(e){const t=document.getElementById("doc-list");t&&(t.innerHTML=e.map(i=>j(i)).join(""),t.querySelectorAll(".logis-result").forEach(i=>{i.addEventListener("click",()=>{a==null||a.send(JSON.stringify({type:"get_detail",uuid:i.id}))})}))}function Z(e,t){document.getElementById("detail-title").innerText=e,document.getElementById("detail-content").innerHTML=t,H(!0)}function R(e){const t=document.createElement("div");t.className=`chat-talk ${e.role==="user"?"user":"system"}`,t.innerHTML=`<div class="chat-message"><div class="content">${e.content}</div></div>`,U.appendChild(t),O.scrollTop=O.scrollHeight}h==null||h.addEventListener("input",e=>{a==null||a.send(JSON.stringify({type:"search",query:e.target.value}))});w==null||w.addEventListener("submit",e=>{e.preventDefault();const t=w.querySelector('input[name="talk"]');t.value.trim()&&(R({role:"user",content:t.value}),a==null||a.send(JSON.stringify({type:"chat_message",content:t.value})),t.value="")});var T;(T=document.getElementById("btn-settings"))==null||T.addEventListener("click",()=>A("settings"));var B;(B=document.getElementById("btn-settings-back"))==null||B.addEventListener("click",()=>A("list"));var C;(C=document.getElementById("btn-detail-back"))==null||C.addEventListener("click",()=>H(!1));var D;(D=document.getElementById("list-refresh-btn"))==null||D.addEventListener("click",()=>{a==null||a.send(JSON.stringify({type:"search",query:(h==null?void 0:h.value)||""}))});window.addEventListener("request-camera-start",()=>F());window.addEventListener("request-camera-stop",()=>P());
