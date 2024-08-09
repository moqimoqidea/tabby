"use strict";(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[2855],{53241:function(e,t,r){r.d(t,{mY:function(){return Z}});var n=/[\\\/_+.#"@\[\(\{&]/,l=/[\\\/_+.#"@\[\(\{&]/g,u=/[\s-]/,a=/[\s-]/g;function i(e){return e.toLowerCase().replace(a," ")}var o=r(4318),c=r(3546),d=r(72205),s='[cmdk-group=""]',f='[cmdk-group-items=""]',v='[cmdk-item=""]',m=`${v}:not([aria-disabled="true"])`,p="cmdk-item-select",h="data-value",g=(e,t,r)=>{var o;return o=e,function e(t,r,i,o,c,d,s){if(d===r.length)return c===t.length?1:.99;var f=`${c},${d}`;if(void 0!==s[f])return s[f];for(var v,m,p,h,g=o.charAt(d),b=i.indexOf(g,c),E=0;b>=0;)(v=e(t,r,i,o,b+1,d+1,s))>E&&(b===c?v*=1:n.test(t.charAt(b-1))?(v*=.8,(p=t.slice(c,b-1).match(l))&&c>0&&(v*=Math.pow(.999,p.length))):u.test(t.charAt(b-1))?(v*=.9,(h=t.slice(c,b-1).match(a))&&c>0&&(v*=Math.pow(.999,h.length))):(v*=.17,c>0&&(v*=Math.pow(.999,b-c))),t.charAt(b)!==r.charAt(d)&&(v*=.9999)),(v<.1&&i.charAt(b-1)===o.charAt(d+1)||o.charAt(d+1)===o.charAt(d)&&i.charAt(b-1)!==o.charAt(d))&&.1*(m=e(t,r,i,o,b+1,d+2,s))>v&&(v=.1*m),v>E&&(E=v),b=i.indexOf(g,b+1);return s[f]=E,E}(o=r&&r.length>0?`${o+" "+r.join(" ")}`:o,t,i(o),i(t),0,0,{})},b=c.createContext(void 0),E=()=>c.useContext(b),w=c.createContext(void 0),y=()=>c.useContext(w),k=c.createContext(void 0),S=c.forwardRef((e,t)=>{let r=W(()=>{var t,r;return{search:"",value:null!=(r=null!=(t=e.value)?t:e.defaultValue)?r:"",filtered:{count:0,items:new Map,groups:new Set}}}),n=W(()=>new Set),l=W(()=>new Map),u=W(()=>new Map),a=W(()=>new Set),i=V(e),{label:o,children:E,value:y,onValueChange:k,filter:S,shouldFilter:C,loop:x,disablePointerSelection:A=!1,vimBindings:I=!0,...R}=e,M=c.useId(),Z=c.useId(),D=c.useId(),P=c.useRef(null),O=j();$(()=>{if(void 0!==y){let e=y.trim();r.current.value=e,q.emit()}},[y]),$(()=>{O(6,U)},[]);let q=c.useMemo(()=>({subscribe:e=>(a.current.add(e),()=>a.current.delete(e)),snapshot:()=>r.current,setState:(e,t,n)=>{var l,u,a;if(!Object.is(r.current[e],t)){if(r.current[e]=t,"search"===e)z(),_(),O(1,N);else if("value"===e&&(n||O(5,U),(null==(l=i.current)?void 0:l.value)!==void 0)){let e=null!=t?t:"";null==(a=(u=i.current).onValueChange)||a.call(u,e);return}q.emit()}},emit:()=>{a.current.forEach(e=>e())}}),[]),L=c.useMemo(()=>({value:(e,t,n)=>{var l;t!==(null==(l=u.current.get(e))?void 0:l.value)&&(u.current.set(e,{value:t,keywords:n}),r.current.filtered.items.set(e,T(t,n)),O(2,()=>{_(),q.emit()}))},item:(e,t)=>(n.current.add(e),t&&(l.current.has(t)?l.current.get(t).add(e):l.current.set(t,new Set([e]))),O(3,()=>{z(),_(),r.current.value||N(),q.emit()}),()=>{u.current.delete(e),n.current.delete(e),r.current.filtered.items.delete(e);let t=H();O(4,()=>{z(),(null==t?void 0:t.getAttribute("id"))===e&&N(),q.emit()})}),group:e=>(l.current.has(e)||l.current.set(e,new Set),()=>{u.current.delete(e),l.current.delete(e)}),filter:()=>i.current.shouldFilter,label:o||e["aria-label"],disablePointerSelection:A,listId:M,inputId:D,labelId:Z,listInnerRef:P}),[]);function T(e,t){var n,l;let u=null!=(l=null==(n=i.current)?void 0:n.filter)?l:g;return e?u(e,r.current.search,t):0}function _(){if(!r.current.search||!1===i.current.shouldFilter)return;let e=r.current.filtered.items,t=[];r.current.filtered.groups.forEach(r=>{let n=l.current.get(r),u=0;n.forEach(t=>{u=Math.max(e.get(t),u)}),t.push([r,u])});let n=P.current;Y().sort((t,r)=>{var n,l;let u=t.getAttribute("id"),a=r.getAttribute("id");return(null!=(n=e.get(a))?n:0)-(null!=(l=e.get(u))?l:0)}).forEach(e=>{let t=e.closest(f);t?t.appendChild(e.parentElement===t?e:e.closest(`${f} > *`)):n.appendChild(e.parentElement===n?e:e.closest(`${f} > *`))}),t.sort((e,t)=>t[1]-e[1]).forEach(e=>{let t=P.current.querySelector(`${s}[${h}="${encodeURIComponent(e[0])}"]`);null==t||t.parentElement.appendChild(t)})}function N(){let e=Y().find(e=>"true"!==e.getAttribute("aria-disabled")),t=null==e?void 0:e.getAttribute(h);q.setState("value",t||void 0)}function z(){var e,t,a,o;if(!r.current.search||!1===i.current.shouldFilter){r.current.filtered.count=n.current.size;return}r.current.filtered.groups=new Set;let c=0;for(let l of n.current){let n=T(null!=(t=null==(e=u.current.get(l))?void 0:e.value)?t:"",null!=(o=null==(a=u.current.get(l))?void 0:a.keywords)?o:[]);r.current.filtered.items.set(l,n),n>0&&c++}for(let[e,t]of l.current)for(let n of t)if(r.current.filtered.items.get(n)>0){r.current.filtered.groups.add(e);break}r.current.filtered.count=c}function U(){var e,t,r;let n=H();n&&((null==(e=n.parentElement)?void 0:e.firstChild)===n&&(null==(r=null==(t=n.closest(s))?void 0:t.querySelector('[cmdk-group-heading=""]'))||r.scrollIntoView({block:"nearest"})),n.scrollIntoView({block:"nearest"}))}function H(){var e;return null==(e=P.current)?void 0:e.querySelector(`${v}[aria-selected="true"]`)}function Y(){var e;return Array.from(null==(e=P.current)?void 0:e.querySelectorAll(m))}function B(e){let t=Y()[e];t&&q.setState("value",t.getAttribute(h))}function G(e){var t;let r=H(),n=Y(),l=n.findIndex(e=>e===r),u=n[l+e];null!=(t=i.current)&&t.loop&&(u=l+e<0?n[n.length-1]:l+e===n.length?n[0]:n[l+e]),u&&q.setState("value",u.getAttribute(h))}function J(e){let t=H(),r=null==t?void 0:t.closest(s),n;for(;r&&!n;)n=null==(r=e>0?function(e,t){let r=e.nextElementSibling;for(;r;){if(r.matches(t))return r;r=r.nextElementSibling}}(r,s):function(e,t){let r=e.previousElementSibling;for(;r;){if(r.matches(t))return r;r=r.previousElementSibling}}(r,s))?void 0:r.querySelector(m);n?q.setState("value",n.getAttribute(h)):G(e)}let Q=()=>B(Y().length-1),X=e=>{e.preventDefault(),e.metaKey?Q():e.altKey?J(1):G(1)},ee=e=>{e.preventDefault(),e.metaKey?B(0):e.altKey?J(-1):G(-1)};return c.createElement(d.WV.div,{ref:t,tabIndex:-1,...R,"cmdk-root":"",onKeyDown:e=>{var t;if(null==(t=R.onKeyDown)||t.call(R,e),!e.defaultPrevented)switch(e.key){case"n":case"j":I&&e.ctrlKey&&X(e);break;case"ArrowDown":X(e);break;case"p":case"k":I&&e.ctrlKey&&ee(e);break;case"ArrowUp":ee(e);break;case"Home":e.preventDefault(),B(0);break;case"End":e.preventDefault(),Q();break;case"Enter":if(!e.nativeEvent.isComposing&&229!==e.keyCode){e.preventDefault();let t=H();if(t){let e=new Event(p);t.dispatchEvent(e)}}}}},c.createElement("label",{"cmdk-label":"",htmlFor:L.inputId,id:L.labelId,style:K},o),F(e,e=>c.createElement(w.Provider,{value:q},c.createElement(b.Provider,{value:L},e))))}),C=c.forwardRef((e,t)=>{var r,n;let l=c.useId(),u=c.useRef(null),a=c.useContext(k),i=E(),o=V(e),s=null!=(n=null==(r=o.current)?void 0:r.forceMount)?n:null==a?void 0:a.forceMount;$(()=>{if(!s)return i.item(l,null==a?void 0:a.id)},[s]);let f=O(l,u,[e.value,e.children,u],e.keywords),v=y(),m=P(e=>e.value&&e.value===f.current),h=P(e=>!!s||!1===i.filter()||!e.search||e.filtered.items.get(l)>0);function g(){var e,t;b(),null==(t=(e=o.current).onSelect)||t.call(e,f.current)}function b(){v.setState("value",f.current,!0)}if(c.useEffect(()=>{let t=u.current;if(!(!t||e.disabled))return t.addEventListener(p,g),()=>t.removeEventListener(p,g)},[h,e.onSelect,e.disabled]),!h)return null;let{disabled:w,value:S,onSelect:C,forceMount:x,keywords:A,...I}=e;return c.createElement(d.WV.div,{ref:D([u,t]),...I,id:l,"cmdk-item":"",role:"option","aria-disabled":!!w,"aria-selected":!!m,"data-disabled":!!w,"data-selected":!!m,onPointerMove:w||i.disablePointerSelection?void 0:b,onClick:w?void 0:g},e.children)}),x=c.forwardRef((e,t)=>{let{heading:r,children:n,forceMount:l,...u}=e,a=c.useId(),i=c.useRef(null),o=c.useRef(null),s=c.useId(),f=E(),v=P(e=>!!l||!1===f.filter()||!e.search||e.filtered.groups.has(a));$(()=>f.group(a),[]),O(a,i,[e.value,e.heading,o]);let m=c.useMemo(()=>({id:a,forceMount:l}),[l]);return c.createElement(d.WV.div,{ref:D([i,t]),...u,"cmdk-group":"",role:"presentation",hidden:!v||void 0},r&&c.createElement("div",{ref:o,"cmdk-group-heading":"","aria-hidden":!0,id:s},r),F(e,e=>c.createElement("div",{"cmdk-group-items":"",role:"group","aria-labelledby":r?s:void 0},c.createElement(k.Provider,{value:m},e))))}),A=c.forwardRef((e,t)=>{let{alwaysRender:r,...n}=e,l=c.useRef(null),u=P(e=>!e.search);return r||u?c.createElement(d.WV.div,{ref:D([l,t]),...n,"cmdk-separator":"",role:"separator"}):null}),I=c.forwardRef((e,t)=>{let{onValueChange:r,...n}=e,l=null!=e.value,u=y(),a=P(e=>e.search),i=P(e=>e.value),o=E(),s=c.useMemo(()=>{var e;let t=null==(e=o.listInnerRef.current)?void 0:e.querySelector(`${v}[${h}="${encodeURIComponent(i)}"]`);return null==t?void 0:t.getAttribute("id")},[]);return c.useEffect(()=>{null!=e.value&&u.setState("search",e.value)},[e.value]),c.createElement(d.WV.input,{ref:t,...n,"cmdk-input":"",autoComplete:"off",autoCorrect:"off",spellCheck:!1,"aria-autocomplete":"list",role:"combobox","aria-expanded":!0,"aria-controls":o.listId,"aria-labelledby":o.labelId,"aria-activedescendant":s,id:o.inputId,type:"text",value:l?e.value:a,onChange:e=>{l||u.setState("search",e.target.value),null==r||r(e.target.value)}})}),R=c.forwardRef((e,t)=>{let{children:r,label:n="Suggestions",...l}=e,u=c.useRef(null),a=c.useRef(null),i=E();return c.useEffect(()=>{if(a.current&&u.current){let e=a.current,t=u.current,r,n=new ResizeObserver(()=>{r=requestAnimationFrame(()=>{let r=e.offsetHeight;t.style.setProperty("--cmdk-list-height",r.toFixed(1)+"px")})});return n.observe(e),()=>{cancelAnimationFrame(r),n.unobserve(e)}}},[]),c.createElement(d.WV.div,{ref:D([u,t]),...l,"cmdk-list":"",role:"listbox","aria-label":n,id:i.listId},F(e,e=>c.createElement("div",{ref:D([a,i.listInnerRef]),"cmdk-list-sizer":""},e)))}),M=c.forwardRef((e,t)=>{let{open:r,onOpenChange:n,overlayClassName:l,contentClassName:u,container:a,...i}=e;return c.createElement(o.fC,{open:r,onOpenChange:n},c.createElement(o.h_,{container:a},c.createElement(o.aV,{"cmdk-overlay":"",className:l}),c.createElement(o.VY,{"aria-label":e.label,"cmdk-dialog":"",className:u},c.createElement(S,{ref:t,...i}))))}),Z=Object.assign(S,{List:R,Item:C,Input:I,Group:x,Separator:A,Dialog:M,Empty:c.forwardRef((e,t)=>P(e=>0===e.filtered.count)?c.createElement(d.WV.div,{ref:t,...e,"cmdk-empty":"",role:"presentation"}):null),Loading:c.forwardRef((e,t)=>{let{progress:r,children:n,label:l="Loading...",...u}=e;return c.createElement(d.WV.div,{ref:t,...u,"cmdk-loading":"",role:"progressbar","aria-valuenow":r,"aria-valuemin":0,"aria-valuemax":100,"aria-label":l},F(e,e=>c.createElement("div",{"aria-hidden":!0},e)))})});function V(e){let t=c.useRef(e);return $(()=>{t.current=e}),t}var $="undefined"==typeof window?c.useEffect:c.useLayoutEffect;function W(e){let t=c.useRef();return void 0===t.current&&(t.current=e()),t}function D(e){return t=>{e.forEach(e=>{"function"==typeof e?e(t):null!=e&&(e.current=t)})}}function P(e){let t=y(),r=()=>e(t.snapshot());return c.useSyncExternalStore(t.subscribe,r,r)}function O(e,t,r,n=[]){let l=c.useRef(),u=E();return $(()=>{var a;let i=(()=>{var e;for(let t of r){if("string"==typeof t)return t.trim();if("object"==typeof t&&"current"in t)return t.current?null==(e=t.current.textContent)?void 0:e.trim():l.current}})(),o=n.map(e=>e.trim());u.value(e,i,o),null==(a=t.current)||a.setAttribute(h,i),l.current=i}),l}var j=()=>{let[e,t]=c.useState(),r=W(()=>new Map);return $(()=>{r.current.forEach(e=>e()),r.current=new Map},[e]),(e,n)=>{r.current.set(e,n),t({})}};function F({asChild:e,children:t},r){let n;return e&&c.isValidElement(t)?c.cloneElement("function"==typeof(n=t.type)?n(t.props):"render"in n?n.render(t.props):t,{ref:t.ref},r(t.props.children)):r(t)}var K={position:"absolute",width:"1px",height:"1px",padding:"0",margin:"-1px",overflow:"hidden",clip:"rect(0, 0, 0, 0)",whiteSpace:"nowrap",borderWidth:"0"}},96703:function(e,t){t.Z=function(e,t,r,n){for(var l=e.length,u=r+(n?1:-1);n?u--:++u<l;)if(t(e[u],u,e))return u;return -1}},45391:function(e,t,r){r.d(t,{Z:function(){return c}});var n=r(84639),l=r(48717),u=function(){return l.Z.Date.now()},a=r(26165),i=Math.max,o=Math.min,c=function(e,t,r){var l,c,d,s,f,v,m=0,p=!1,h=!1,g=!0;if("function"!=typeof e)throw TypeError("Expected a function");function b(t){var r=l,n=c;return l=c=void 0,m=t,s=e.apply(n,r)}function E(e){var r=e-v,n=e-m;return void 0===v||r>=t||r<0||h&&n>=d}function w(){var e,r,n,l=u();if(E(l))return y(l);f=setTimeout(w,(e=l-v,r=l-m,n=t-e,h?o(n,d-r):n))}function y(e){return(f=void 0,g&&l)?b(e):(l=c=void 0,s)}function k(){var e,r=u(),n=E(r);if(l=arguments,c=this,v=r,n){if(void 0===f)return m=e=v,f=setTimeout(w,t),p?b(e):s;if(h)return clearTimeout(f),f=setTimeout(w,t),b(v)}return void 0===f&&(f=setTimeout(w,t)),s}return t=(0,a.Z)(t)||0,(0,n.Z)(r)&&(p=!!r.leading,d=(h="maxWait"in r)?i((0,a.Z)(r.maxWait)||0,t):d,g="trailing"in r?!!r.trailing:g),k.cancel=function(){void 0!==f&&clearTimeout(f),m=0,l=v=c=f=void 0},k.flush=function(){return void 0===f?s:y(u())},k}},78613:function(e,t,r){r.d(t,{Z:function(){return c}});var n=r(96703),l=r(12894),u=r(26165),a=1/0,i=function(e){var t,r=(t=e)?(t=(0,u.Z)(t))===a||t===-a?(t<0?-1:1)*17976931348623157e292:t==t?t:0:0===t?t:0,n=r%1;return r==r?n?r-n:r:0},o=Math.max,c=function(e,t,r){var u=null==e?0:e.length;if(!u)return -1;var a=null==r?0:i(r);return a<0&&(a=o(u+a,0)),(0,n.Z)(e,(0,l.Z)(t,3),a)}},94909:function(e,t,r){var n=r(63563),l=r(64134),u=r(97589),a=r(38813),i=r(20568),o=r(90328),c=r(36586),d=r(33321),s=Object.prototype.hasOwnProperty;t.Z=function(e){if(null==e)return!0;if((0,i.Z)(e)&&((0,a.Z)(e)||"string"==typeof e||"function"==typeof e.splice||(0,o.Z)(e)||(0,d.Z)(e)||(0,u.Z)(e)))return!e.length;var t=(0,l.Z)(e);if("[object Map]"==t||"[object Set]"==t)return!e.size;if((0,c.Z)(e))return!(0,n.Z)(e).length;for(var r in e)if(s.call(e,r))return!1;return!0}}}]);