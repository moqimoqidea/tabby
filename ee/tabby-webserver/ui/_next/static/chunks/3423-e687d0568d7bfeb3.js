(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[3423],{70652:function(e,t,r){e.exports=r(54007)},96887:function(e,t,r){"use strict";r.d(t,{bU:function(){return y},fC:function(){return k}});var n=r(65122),l=r(3546),a=r(65727),u=r(79869),i=r(47091),c=r(27250),o=r(81544),d=r(96593),s=r(72205);let f="Switch",[v,p]=(0,i.b)(f),[m,h]=v(f),b=(0,l.forwardRef)((e,t)=>{let{__scopeSwitch:r,name:i,checked:o,defaultChecked:d,required:f,disabled:v,value:p="on",onCheckedChange:h,...b}=e,[g,k]=(0,l.useState)(null),y=(0,u.e)(t,e=>k(e)),S=(0,l.useRef)(!1),C=!g||!!g.closest("form"),[R=!1,x]=(0,c.T)({prop:o,defaultProp:d,onChange:h});return(0,l.createElement)(m,{scope:r,checked:R,disabled:v},(0,l.createElement)(s.WV.button,(0,n.Z)({type:"button",role:"switch","aria-checked":R,"aria-required":f,"data-state":w(R),"data-disabled":v?"":void 0,disabled:v,value:p},b,{ref:y,onClick:(0,a.M)(e.onClick,e=>{x(e=>!e),C&&(S.current=e.isPropagationStopped(),S.current||e.stopPropagation())})})),C&&(0,l.createElement)(E,{control:g,bubbles:!S.current,name:i,value:p,checked:R,required:f,disabled:v,style:{transform:"translateX(-100%)"}}))}),g=(0,l.forwardRef)((e,t)=>{let{__scopeSwitch:r,...a}=e,u=h("SwitchThumb",r);return(0,l.createElement)(s.WV.span,(0,n.Z)({"data-state":w(u.checked),"data-disabled":u.disabled?"":void 0},a,{ref:t}))}),E=e=>{let{control:t,checked:r,bubbles:a=!0,...u}=e,i=(0,l.useRef)(null),c=(0,o.D)(r),s=(0,d.t)(t);return(0,l.useEffect)(()=>{let e=i.current,t=window.HTMLInputElement.prototype,n=Object.getOwnPropertyDescriptor(t,"checked"),l=n.set;if(c!==r&&l){let t=new Event("click",{bubbles:a});l.call(e,r),e.dispatchEvent(t)}},[c,r,a]),(0,l.createElement)("input",(0,n.Z)({type:"checkbox","aria-hidden":!0,defaultChecked:r},u,{tabIndex:-1,ref:i,style:{...e.style,...s,position:"absolute",pointerEvents:"none",opacity:0,margin:0}}))};function w(e){return e?"checked":"unchecked"}let k=b,y=g},53241:function(e,t,r){"use strict";r.d(t,{mY:function(){return V}});var n=/[\\\/_+.#"@\[\(\{&]/,l=/[\\\/_+.#"@\[\(\{&]/g,a=/[\s-]/,u=/[\s-]/g;function i(e){return e.toLowerCase().replace(u," ")}var c=r(4318),o=r(3546),d=r(72205),s='[cmdk-group=""]',f='[cmdk-group-items=""]',v='[cmdk-item=""]',p=`${v}:not([aria-disabled="true"])`,m="cmdk-item-select",h="data-value",b=(e,t,r)=>{var c;return c=e,function e(t,r,i,c,o,d,s){if(d===r.length)return o===t.length?1:.99;var f=`${o},${d}`;if(void 0!==s[f])return s[f];for(var v,p,m,h,b=c.charAt(d),g=i.indexOf(b,o),E=0;g>=0;)(v=e(t,r,i,c,g+1,d+1,s))>E&&(g===o?v*=1:n.test(t.charAt(g-1))?(v*=.8,(m=t.slice(o,g-1).match(l))&&o>0&&(v*=Math.pow(.999,m.length))):a.test(t.charAt(g-1))?(v*=.9,(h=t.slice(o,g-1).match(u))&&o>0&&(v*=Math.pow(.999,h.length))):(v*=.17,o>0&&(v*=Math.pow(.999,g-o))),t.charAt(g)!==r.charAt(d)&&(v*=.9999)),(v<.1&&i.charAt(g-1)===c.charAt(d+1)||c.charAt(d+1)===c.charAt(d)&&i.charAt(g-1)!==c.charAt(d))&&.1*(p=e(t,r,i,c,g+1,d+2,s))>v&&(v=.1*p),v>E&&(E=v),g=i.indexOf(b,g+1);return s[f]=E,E}(c=r&&r.length>0?`${c+" "+r.join(" ")}`:c,t,i(c),i(t),0,0,{})},g=o.createContext(void 0),E=()=>o.useContext(g),w=o.createContext(void 0),k=()=>o.useContext(w),y=o.createContext(void 0),S=o.forwardRef((e,t)=>{let r=P(()=>{var t,r;return{search:"",value:null!=(r=null!=(t=e.value)?t:e.defaultValue)?r:"",filtered:{count:0,items:new Map,groups:new Set}}}),n=P(()=>new Set),l=P(()=>new Map),a=P(()=>new Map),u=P(()=>new Set),i=$(e),{label:c,children:E,value:k,onValueChange:y,filter:S,shouldFilter:C,loop:R,disablePointerSelection:x=!1,vimBindings:I=!0,...A}=e,M=o.useId(),V=o.useId(),D=o.useId(),O=o.useRef(null),W=_();Z(()=>{if(void 0!==k){let e=k.trim();r.current.value=e,K.emit()}},[k]),Z(()=>{W(6,H)},[]);let K=o.useMemo(()=>({subscribe:e=>(u.current.add(e),()=>u.current.delete(e)),snapshot:()=>r.current,setState:(e,t,n)=>{var l,a,u;if(!Object.is(r.current[e],t)){if(r.current[e]=t,"search"===e)z(),N(),W(1,U);else if("value"===e&&(n||W(5,H),(null==(l=i.current)?void 0:l.value)!==void 0)){let e=null!=t?t:"";null==(u=(a=i.current).onValueChange)||u.call(a,e);return}K.emit()}},emit:()=>{u.current.forEach(e=>e())}}),[]),L=o.useMemo(()=>({value:(e,t,n)=>{var l;t!==(null==(l=a.current.get(e))?void 0:l.value)&&(a.current.set(e,{value:t,keywords:n}),r.current.filtered.items.set(e,j(t,n)),W(2,()=>{N(),K.emit()}))},item:(e,t)=>(n.current.add(e),t&&(l.current.has(t)?l.current.get(t).add(e):l.current.set(t,new Set([e]))),W(3,()=>{z(),N(),r.current.value||U(),K.emit()}),()=>{a.current.delete(e),n.current.delete(e),r.current.filtered.items.delete(e);let t=T();W(4,()=>{z(),(null==t?void 0:t.getAttribute("id"))===e&&U(),K.emit()})}),group:e=>(l.current.has(e)||l.current.set(e,new Set),()=>{a.current.delete(e),l.current.delete(e)}),filter:()=>i.current.shouldFilter,label:c||e["aria-label"],disablePointerSelection:x,listId:M,inputId:D,labelId:V,listInnerRef:O}),[]);function j(e,t){var n,l;let a=null!=(l=null==(n=i.current)?void 0:n.filter)?l:b;return e?a(e,r.current.search,t):0}function N(){if(!r.current.search||!1===i.current.shouldFilter)return;let e=r.current.filtered.items,t=[];r.current.filtered.groups.forEach(r=>{let n=l.current.get(r),a=0;n.forEach(t=>{a=Math.max(e.get(t),a)}),t.push([r,a])});let n=O.current;Y().sort((t,r)=>{var n,l;let a=t.getAttribute("id"),u=r.getAttribute("id");return(null!=(n=e.get(u))?n:0)-(null!=(l=e.get(a))?l:0)}).forEach(e=>{let t=e.closest(f);t?t.appendChild(e.parentElement===t?e:e.closest(`${f} > *`)):n.appendChild(e.parentElement===n?e:e.closest(`${f} > *`))}),t.sort((e,t)=>t[1]-e[1]).forEach(e=>{let t=O.current.querySelector(`${s}[${h}="${encodeURIComponent(e[0])}"]`);null==t||t.parentElement.appendChild(t)})}function U(){let e=Y().find(e=>"true"!==e.getAttribute("aria-disabled")),t=null==e?void 0:e.getAttribute(h);K.setState("value",t||void 0)}function z(){var e,t,u,c;if(!r.current.search||!1===i.current.shouldFilter){r.current.filtered.count=n.current.size;return}r.current.filtered.groups=new Set;let o=0;for(let l of n.current){let n=j(null!=(t=null==(e=a.current.get(l))?void 0:e.value)?t:"",null!=(c=null==(u=a.current.get(l))?void 0:u.keywords)?c:[]);r.current.filtered.items.set(l,n),n>0&&o++}for(let[e,t]of l.current)for(let n of t)if(r.current.filtered.items.get(n)>0){r.current.filtered.groups.add(e);break}r.current.filtered.count=o}function H(){var e,t,r;let n=T();n&&((null==(e=n.parentElement)?void 0:e.firstChild)===n&&(null==(r=null==(t=n.closest(s))?void 0:t.querySelector('[cmdk-group-heading=""]'))||r.scrollIntoView({block:"nearest"})),n.scrollIntoView({block:"nearest"}))}function T(){var e;return null==(e=O.current)?void 0:e.querySelector(`${v}[aria-selected="true"]`)}function Y(){var e;return Array.from(null==(e=O.current)?void 0:e.querySelectorAll(p))}function B(e){let t=Y()[e];t&&K.setState("value",t.getAttribute(h))}function G(e){var t;let r=T(),n=Y(),l=n.findIndex(e=>e===r),a=n[l+e];null!=(t=i.current)&&t.loop&&(a=l+e<0?n[n.length-1]:l+e===n.length?n[0]:n[l+e]),a&&K.setState("value",a.getAttribute(h))}function X(e){let t=T(),r=null==t?void 0:t.closest(s),n;for(;r&&!n;)n=null==(r=e>0?function(e,t){let r=e.nextElementSibling;for(;r;){if(r.matches(t))return r;r=r.nextElementSibling}}(r,s):function(e,t){let r=e.previousElementSibling;for(;r;){if(r.matches(t))return r;r=r.previousElementSibling}}(r,s))?void 0:r.querySelector(p);n?K.setState("value",n.getAttribute(h)):G(e)}let J=()=>B(Y().length-1),Q=e=>{e.preventDefault(),e.metaKey?J():e.altKey?X(1):G(1)},ee=e=>{e.preventDefault(),e.metaKey?B(0):e.altKey?X(-1):G(-1)};return o.createElement(d.WV.div,{ref:t,tabIndex:-1,...A,"cmdk-root":"",onKeyDown:e=>{var t;if(null==(t=A.onKeyDown)||t.call(A,e),!e.defaultPrevented)switch(e.key){case"n":case"j":I&&e.ctrlKey&&Q(e);break;case"ArrowDown":Q(e);break;case"p":case"k":I&&e.ctrlKey&&ee(e);break;case"ArrowUp":ee(e);break;case"Home":e.preventDefault(),B(0);break;case"End":e.preventDefault(),J();break;case"Enter":if(!e.nativeEvent.isComposing&&229!==e.keyCode){e.preventDefault();let t=T();if(t){let e=new Event(m);t.dispatchEvent(e)}}}}},o.createElement("label",{"cmdk-label":"",htmlFor:L.inputId,id:L.labelId,style:F},c),q(e,e=>o.createElement(w.Provider,{value:K},o.createElement(g.Provider,{value:L},e))))}),C=o.forwardRef((e,t)=>{var r,n;let l=o.useId(),a=o.useRef(null),u=o.useContext(y),i=E(),c=$(e),s=null!=(n=null==(r=c.current)?void 0:r.forceMount)?n:null==u?void 0:u.forceMount;Z(()=>{if(!s)return i.item(l,null==u?void 0:u.id)},[s]);let f=W(l,a,[e.value,e.children,a],e.keywords),v=k(),p=O(e=>e.value&&e.value===f.current),h=O(e=>!!s||!1===i.filter()||!e.search||e.filtered.items.get(l)>0);function b(){var e,t;g(),null==(t=(e=c.current).onSelect)||t.call(e,f.current)}function g(){v.setState("value",f.current,!0)}if(o.useEffect(()=>{let t=a.current;if(!(!t||e.disabled))return t.addEventListener(m,b),()=>t.removeEventListener(m,b)},[h,e.onSelect,e.disabled]),!h)return null;let{disabled:w,value:S,onSelect:C,forceMount:R,keywords:x,...I}=e;return o.createElement(d.WV.div,{ref:D([a,t]),...I,id:l,"cmdk-item":"",role:"option","aria-disabled":!!w,"aria-selected":!!p,"data-disabled":!!w,"data-selected":!!p,onPointerMove:w||i.disablePointerSelection?void 0:g,onClick:w?void 0:b},e.children)}),R=o.forwardRef((e,t)=>{let{heading:r,children:n,forceMount:l,...a}=e,u=o.useId(),i=o.useRef(null),c=o.useRef(null),s=o.useId(),f=E(),v=O(e=>!!l||!1===f.filter()||!e.search||e.filtered.groups.has(u));Z(()=>f.group(u),[]),W(u,i,[e.value,e.heading,c]);let p=o.useMemo(()=>({id:u,forceMount:l}),[l]);return o.createElement(d.WV.div,{ref:D([i,t]),...a,"cmdk-group":"",role:"presentation",hidden:!v||void 0},r&&o.createElement("div",{ref:c,"cmdk-group-heading":"","aria-hidden":!0,id:s},r),q(e,e=>o.createElement("div",{"cmdk-group-items":"",role:"group","aria-labelledby":r?s:void 0},o.createElement(y.Provider,{value:p},e))))}),x=o.forwardRef((e,t)=>{let{alwaysRender:r,...n}=e,l=o.useRef(null),a=O(e=>!e.search);return r||a?o.createElement(d.WV.div,{ref:D([l,t]),...n,"cmdk-separator":"",role:"separator"}):null}),I=o.forwardRef((e,t)=>{let{onValueChange:r,...n}=e,l=null!=e.value,a=k(),u=O(e=>e.search),i=O(e=>e.value),c=E(),s=o.useMemo(()=>{var e;let t=null==(e=c.listInnerRef.current)?void 0:e.querySelector(`${v}[${h}="${encodeURIComponent(i)}"]`);return null==t?void 0:t.getAttribute("id")},[]);return o.useEffect(()=>{null!=e.value&&a.setState("search",e.value)},[e.value]),o.createElement(d.WV.input,{ref:t,...n,"cmdk-input":"",autoComplete:"off",autoCorrect:"off",spellCheck:!1,"aria-autocomplete":"list",role:"combobox","aria-expanded":!0,"aria-controls":c.listId,"aria-labelledby":c.labelId,"aria-activedescendant":s,id:c.inputId,type:"text",value:l?e.value:u,onChange:e=>{l||a.setState("search",e.target.value),null==r||r(e.target.value)}})}),A=o.forwardRef((e,t)=>{let{children:r,label:n="Suggestions",...l}=e,a=o.useRef(null),u=o.useRef(null),i=E();return o.useEffect(()=>{if(u.current&&a.current){let e=u.current,t=a.current,r,n=new ResizeObserver(()=>{r=requestAnimationFrame(()=>{let r=e.offsetHeight;t.style.setProperty("--cmdk-list-height",r.toFixed(1)+"px")})});return n.observe(e),()=>{cancelAnimationFrame(r),n.unobserve(e)}}},[]),o.createElement(d.WV.div,{ref:D([a,t]),...l,"cmdk-list":"",role:"listbox","aria-label":n,id:i.listId},q(e,e=>o.createElement("div",{ref:D([u,i.listInnerRef]),"cmdk-list-sizer":""},e)))}),M=o.forwardRef((e,t)=>{let{open:r,onOpenChange:n,overlayClassName:l,contentClassName:a,container:u,...i}=e;return o.createElement(c.fC,{open:r,onOpenChange:n},o.createElement(c.h_,{container:u},o.createElement(c.aV,{"cmdk-overlay":"",className:l}),o.createElement(c.VY,{"aria-label":e.label,"cmdk-dialog":"",className:a},o.createElement(S,{ref:t,...i}))))}),V=Object.assign(S,{List:A,Item:C,Input:I,Group:R,Separator:x,Dialog:M,Empty:o.forwardRef((e,t)=>O(e=>0===e.filtered.count)?o.createElement(d.WV.div,{ref:t,...e,"cmdk-empty":"",role:"presentation"}):null),Loading:o.forwardRef((e,t)=>{let{progress:r,children:n,label:l="Loading...",...a}=e;return o.createElement(d.WV.div,{ref:t,...a,"cmdk-loading":"",role:"progressbar","aria-valuenow":r,"aria-valuemin":0,"aria-valuemax":100,"aria-label":l},q(e,e=>o.createElement("div",{"aria-hidden":!0},e)))})});function $(e){let t=o.useRef(e);return Z(()=>{t.current=e}),t}var Z="undefined"==typeof window?o.useEffect:o.useLayoutEffect;function P(e){let t=o.useRef();return void 0===t.current&&(t.current=e()),t}function D(e){return t=>{e.forEach(e=>{"function"==typeof e?e(t):null!=e&&(e.current=t)})}}function O(e){let t=k(),r=()=>e(t.snapshot());return o.useSyncExternalStore(t.subscribe,r,r)}function W(e,t,r,n=[]){let l=o.useRef(),a=E();return Z(()=>{var u;let i=(()=>{var e;for(let t of r){if("string"==typeof t)return t.trim();if("object"==typeof t&&"current"in t)return t.current?null==(e=t.current.textContent)?void 0:e.trim():l.current}})(),c=n.map(e=>e.trim());a.value(e,i,c),null==(u=t.current)||u.setAttribute(h,i),l.current=i}),l}var _=()=>{let[e,t]=o.useState(),r=P(()=>new Map);return Z(()=>{r.current.forEach(e=>e()),r.current=new Map},[e]),(e,n)=>{r.current.set(e,n),t({})}};function q({asChild:e,children:t},r){let n;return e&&o.isValidElement(t)?o.cloneElement("function"==typeof(n=t.type)?n(t.props):"render"in n?n.render(t.props):t,{ref:t.ref},r(t.props.children)):r(t)}var F={position:"absolute",width:"1px",height:"1px",padding:"0",margin:"-1px",overflow:"hidden",clip:"rect(0, 0, 0, 0)",whiteSpace:"nowrap",borderWidth:"0"}},18216:function(e,t,r){"use strict";var n=r(6670),l=/^\s+/;t.Z=function(e){return e?e.slice(0,(0,n.Z)(e)+1).replace(l,""):e}},6670:function(e,t){"use strict";var r=/\s/;t.Z=function(e){for(var t=e.length;t--&&r.test(e.charAt(t)););return t}},55357:function(e,t,r){"use strict";var n=r(17996),l=r(96786);t.Z=function(e){return"symbol"==typeof e||(0,l.Z)(e)&&"[object Symbol]"==(0,n.Z)(e)}},26165:function(e,t,r){"use strict";var n=r(18216),l=r(84639),a=r(55357),u=0/0,i=/^[-+]0x[0-9a-f]+$/i,c=/^0b[01]+$/i,o=/^0o[0-7]+$/i,d=parseInt;t.Z=function(e){if("number"==typeof e)return e;if((0,a.Z)(e))return u;if((0,l.Z)(e)){var t="function"==typeof e.valueOf?e.valueOf():e;e=(0,l.Z)(t)?t+"":t}if("string"!=typeof e)return 0===e?e:+e;e=(0,n.Z)(e);var r=c.test(e);return r||o.test(e)?d(e.slice(2),r?2:8):i.test(e)?u:+e}}}]);