"use strict";(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[3241],{4318:function(e,t,r){r.d(t,{Dx:function(){return B},VY:function(){return Y},aV:function(){return U},dk:function(){return H},fC:function(){return L},h_:function(){return j},x8:function(){return G},xz:function(){return T}});var n=r(65122),l=r(3546),o=r(65727),a=r(79869),u=r(47091),i=r(29434),c=r(27250),d=r(71404),s=r(8914),f=r(48907),p=r(96497),v=r(72205),m=r(12192),g=r(8569),h=r(47847),b=r(74047);let E="Dialog",[w,C]=(0,u.b)(E),[y,R]=w(E),k=(0,l.forwardRef)((e,t)=>{let{__scopeDialog:r,...u}=e,i=R("DialogTrigger",r),c=(0,a.e)(t,i.triggerRef);return(0,l.createElement)(v.WV.button,(0,n.Z)({type:"button","aria-haspopup":"dialog","aria-expanded":i.open,"aria-controls":i.contentId,"data-state":N(i.open)},u,{ref:c,onClick:(0,o.M)(e.onClick,i.onOpenToggle)}))}),D="DialogPortal",[I,S]=w(D,{forceMount:void 0}),M="DialogOverlay",_=(0,l.forwardRef)((e,t)=>{let r=S(M,e.__scopeDialog),{forceMount:o=r.forceMount,...a}=e,u=R(M,e.__scopeDialog);return u.modal?(0,l.createElement)(p.z,{present:o||u.open},(0,l.createElement)(A,(0,n.Z)({},a,{ref:t}))):null}),A=(0,l.forwardRef)((e,t)=>{let{__scopeDialog:r,...o}=e,a=R(M,r);return(0,l.createElement)(g.Z,{as:b.g7,allowPinchZoom:!0,shards:[a.contentRef]},(0,l.createElement)(v.WV.div,(0,n.Z)({"data-state":N(a.open)},o,{ref:t,style:{pointerEvents:"auto",...o.style}})))}),x="DialogContent",O=(0,l.forwardRef)((e,t)=>{let r=S(x,e.__scopeDialog),{forceMount:o=r.forceMount,...a}=e,u=R(x,e.__scopeDialog);return(0,l.createElement)(p.z,{present:o||u.open},u.modal?(0,l.createElement)(V,(0,n.Z)({},a,{ref:t})):(0,l.createElement)(F,(0,n.Z)({},a,{ref:t})))}),V=(0,l.forwardRef)((e,t)=>{let r=R(x,e.__scopeDialog),u=(0,l.useRef)(null),i=(0,a.e)(t,r.contentRef,u);return(0,l.useEffect)(()=>{let e=u.current;if(e)return(0,h.Ry)(e)},[]),(0,l.createElement)(P,(0,n.Z)({},e,{ref:i,trapFocus:r.open,disableOutsidePointerEvents:!0,onCloseAutoFocus:(0,o.M)(e.onCloseAutoFocus,e=>{var t;e.preventDefault(),null===(t=r.triggerRef.current)||void 0===t||t.focus()}),onPointerDownOutside:(0,o.M)(e.onPointerDownOutside,e=>{let t=e.detail.originalEvent,r=0===t.button&&!0===t.ctrlKey,n=2===t.button||r;n&&e.preventDefault()}),onFocusOutside:(0,o.M)(e.onFocusOutside,e=>e.preventDefault())}))}),F=(0,l.forwardRef)((e,t)=>{let r=R(x,e.__scopeDialog),o=(0,l.useRef)(!1),a=(0,l.useRef)(!1);return(0,l.createElement)(P,(0,n.Z)({},e,{ref:t,trapFocus:!1,disableOutsidePointerEvents:!1,onCloseAutoFocus:t=>{var n,l;null===(n=e.onCloseAutoFocus)||void 0===n||n.call(e,t),t.defaultPrevented||(o.current||null===(l=r.triggerRef.current)||void 0===l||l.focus(),t.preventDefault()),o.current=!1,a.current=!1},onInteractOutside:t=>{var n,l;null===(n=e.onInteractOutside)||void 0===n||n.call(e,t),t.defaultPrevented||(o.current=!0,"pointerdown"!==t.detail.originalEvent.type||(a.current=!0));let u=t.target,i=null===(l=r.triggerRef.current)||void 0===l?void 0:l.contains(u);i&&t.preventDefault(),"focusin"===t.detail.originalEvent.type&&a.current&&t.preventDefault()}}))}),P=(0,l.forwardRef)((e,t)=>{let{__scopeDialog:r,trapFocus:o,onOpenAutoFocus:u,onCloseAutoFocus:i,...c}=e,f=R(x,r),p=(0,l.useRef)(null),v=(0,a.e)(t,p);return(0,m.EW)(),(0,l.createElement)(l.Fragment,null,(0,l.createElement)(s.M,{asChild:!0,loop:!0,trapped:o,onMountAutoFocus:u,onUnmountAutoFocus:i},(0,l.createElement)(d.XB,(0,n.Z)({role:"dialog",id:f.contentId,"aria-describedby":f.descriptionId,"aria-labelledby":f.titleId,"data-state":N(f.open)},c,{ref:v,onDismiss:()=>f.onOpenChange(!1)}))),!1)}),W="DialogTitle",Z=(0,l.forwardRef)((e,t)=>{let{__scopeDialog:r,...o}=e,a=R(W,r);return(0,l.createElement)(v.WV.h2,(0,n.Z)({id:a.titleId},o,{ref:t}))}),$=(0,l.forwardRef)((e,t)=>{let{__scopeDialog:r,...o}=e,a=R("DialogDescription",r);return(0,l.createElement)(v.WV.p,(0,n.Z)({id:a.descriptionId},o,{ref:t}))}),K=(0,l.forwardRef)((e,t)=>{let{__scopeDialog:r,...a}=e,u=R("DialogClose",r);return(0,l.createElement)(v.WV.button,(0,n.Z)({type:"button"},a,{ref:t,onClick:(0,o.M)(e.onClick,()=>u.onOpenChange(!1))}))});function N(e){return e?"open":"closed"}let[q,z]=(0,u.k)("DialogTitleWarning",{contentName:x,titleName:W,docsSlug:"dialog"}),L=e=>{let{__scopeDialog:t,children:r,open:n,defaultOpen:o,onOpenChange:a,modal:u=!0}=e,d=(0,l.useRef)(null),s=(0,l.useRef)(null),[f=!1,p]=(0,c.T)({prop:n,defaultProp:o,onChange:a});return(0,l.createElement)(y,{scope:t,triggerRef:d,contentRef:s,contentId:(0,i.M)(),titleId:(0,i.M)(),descriptionId:(0,i.M)(),open:f,onOpenChange:p,onOpenToggle:(0,l.useCallback)(()=>p(e=>!e),[p]),modal:u},r)},T=k,j=e=>{let{__scopeDialog:t,forceMount:r,children:n,container:o}=e,a=R(D,t);return(0,l.createElement)(I,{scope:t,forceMount:r},l.Children.map(n,e=>(0,l.createElement)(p.z,{present:r||a.open},(0,l.createElement)(f.h,{asChild:!0,container:o},e))))},U=_,Y=O,B=Z,H=$,G=K},53241:function(e,t,r){r.d(t,{mY:function(){return A}});var n=/[\\\/_+.#"@\[\(\{&]/,l=/[\\\/_+.#"@\[\(\{&]/g,o=/[\s-]/,a=/[\s-]/g;function u(e){return e.toLowerCase().replace(a," ")}var i=r(4318),c=r(3546),d=r(72205),s='[cmdk-group=""]',f='[cmdk-group-items=""]',p='[cmdk-item=""]',v=`${p}:not([aria-disabled="true"])`,m="cmdk-item-select",g="data-value",h=(e,t,r)=>{var i;return i=e,function e(t,r,u,i,c,d,s){if(d===r.length)return c===t.length?1:.99;var f=`${c},${d}`;if(void 0!==s[f])return s[f];for(var p,v,m,g,h=i.charAt(d),b=u.indexOf(h,c),E=0;b>=0;)(p=e(t,r,u,i,b+1,d+1,s))>E&&(b===c?p*=1:n.test(t.charAt(b-1))?(p*=.8,(m=t.slice(c,b-1).match(l))&&c>0&&(p*=Math.pow(.999,m.length))):o.test(t.charAt(b-1))?(p*=.9,(g=t.slice(c,b-1).match(a))&&c>0&&(p*=Math.pow(.999,g.length))):(p*=.17,c>0&&(p*=Math.pow(.999,b-c))),t.charAt(b)!==r.charAt(d)&&(p*=.9999)),(p<.1&&u.charAt(b-1)===i.charAt(d+1)||i.charAt(d+1)===i.charAt(d)&&u.charAt(b-1)!==i.charAt(d))&&.1*(v=e(t,r,u,i,b+1,d+2,s))>p&&(p=.1*v),p>E&&(E=p),b=u.indexOf(h,b+1);return s[f]=E,E}(i=r&&r.length>0?`${i+" "+r.join(" ")}`:i,t,u(i),u(t),0,0,{})},b=c.createContext(void 0),E=()=>c.useContext(b),w=c.createContext(void 0),C=()=>c.useContext(w),y=c.createContext(void 0),R=c.forwardRef((e,t)=>{let r=V(()=>{var t,r;return{search:"",value:null!=(r=null!=(t=e.value)?t:e.defaultValue)?r:"",filtered:{count:0,items:new Map,groups:new Set}}}),n=V(()=>new Set),l=V(()=>new Map),o=V(()=>new Map),a=V(()=>new Set),u=x(e),{label:i,children:E,value:C,onValueChange:y,filter:R,shouldFilter:k,loop:D,disablePointerSelection:I=!1,vimBindings:S=!0,...M}=e,_=c.useId(),A=c.useId(),F=c.useId(),P=c.useRef(null),W=Z();O(()=>{if(void 0!==C){let e=C.trim();r.current.value=e,N.emit()}},[C]),O(()=>{W(6,U)},[]);let N=c.useMemo(()=>({subscribe:e=>(a.current.add(e),()=>a.current.delete(e)),snapshot:()=>r.current,setState:(e,t,n)=>{var l,o,a;if(!Object.is(r.current[e],t)){if(r.current[e]=t,"search"===e)j(),L(),W(1,T);else if("value"===e&&(n||W(5,U),(null==(l=u.current)?void 0:l.value)!==void 0)){let e=null!=t?t:"";null==(a=(o=u.current).onValueChange)||a.call(o,e);return}N.emit()}},emit:()=>{a.current.forEach(e=>e())}}),[]),q=c.useMemo(()=>({value:(e,t,n)=>{var l;t!==(null==(l=o.current.get(e))?void 0:l.value)&&(o.current.set(e,{value:t,keywords:n}),r.current.filtered.items.set(e,z(t,n)),W(2,()=>{L(),N.emit()}))},item:(e,t)=>(n.current.add(e),t&&(l.current.has(t)?l.current.get(t).add(e):l.current.set(t,new Set([e]))),W(3,()=>{j(),L(),r.current.value||T(),N.emit()}),()=>{o.current.delete(e),n.current.delete(e),r.current.filtered.items.delete(e);let t=Y();W(4,()=>{j(),(null==t?void 0:t.getAttribute("id"))===e&&T(),N.emit()})}),group:e=>(l.current.has(e)||l.current.set(e,new Set),()=>{o.current.delete(e),l.current.delete(e)}),filter:()=>u.current.shouldFilter,label:i||e["aria-label"],disablePointerSelection:I,listId:_,inputId:F,labelId:A,listInnerRef:P}),[]);function z(e,t){var n,l;let o=null!=(l=null==(n=u.current)?void 0:n.filter)?l:h;return e?o(e,r.current.search,t):0}function L(){if(!r.current.search||!1===u.current.shouldFilter)return;let e=r.current.filtered.items,t=[];r.current.filtered.groups.forEach(r=>{let n=l.current.get(r),o=0;n.forEach(t=>{o=Math.max(e.get(t),o)}),t.push([r,o])});let n=P.current;B().sort((t,r)=>{var n,l;let o=t.getAttribute("id"),a=r.getAttribute("id");return(null!=(n=e.get(a))?n:0)-(null!=(l=e.get(o))?l:0)}).forEach(e=>{let t=e.closest(f);t?t.appendChild(e.parentElement===t?e:e.closest(`${f} > *`)):n.appendChild(e.parentElement===n?e:e.closest(`${f} > *`))}),t.sort((e,t)=>t[1]-e[1]).forEach(e=>{let t=P.current.querySelector(`${s}[${g}="${encodeURIComponent(e[0])}"]`);null==t||t.parentElement.appendChild(t)})}function T(){let e=B().find(e=>"true"!==e.getAttribute("aria-disabled")),t=null==e?void 0:e.getAttribute(g);N.setState("value",t||void 0)}function j(){var e,t,a,i;if(!r.current.search||!1===u.current.shouldFilter){r.current.filtered.count=n.current.size;return}r.current.filtered.groups=new Set;let c=0;for(let l of n.current){let n=z(null!=(t=null==(e=o.current.get(l))?void 0:e.value)?t:"",null!=(i=null==(a=o.current.get(l))?void 0:a.keywords)?i:[]);r.current.filtered.items.set(l,n),n>0&&c++}for(let[e,t]of l.current)for(let n of t)if(r.current.filtered.items.get(n)>0){r.current.filtered.groups.add(e);break}r.current.filtered.count=c}function U(){var e,t,r;let n=Y();n&&((null==(e=n.parentElement)?void 0:e.firstChild)===n&&(null==(r=null==(t=n.closest(s))?void 0:t.querySelector('[cmdk-group-heading=""]'))||r.scrollIntoView({block:"nearest"})),n.scrollIntoView({block:"nearest"}))}function Y(){var e;return null==(e=P.current)?void 0:e.querySelector(`${p}[aria-selected="true"]`)}function B(){var e;return Array.from(null==(e=P.current)?void 0:e.querySelectorAll(v))}function H(e){let t=B()[e];t&&N.setState("value",t.getAttribute(g))}function G(e){var t;let r=Y(),n=B(),l=n.findIndex(e=>e===r),o=n[l+e];null!=(t=u.current)&&t.loop&&(o=l+e<0?n[n.length-1]:l+e===n.length?n[0]:n[l+e]),o&&N.setState("value",o.getAttribute(g))}function X(e){let t=Y(),r=null==t?void 0:t.closest(s),n;for(;r&&!n;)n=null==(r=e>0?function(e,t){let r=e.nextElementSibling;for(;r;){if(r.matches(t))return r;r=r.nextElementSibling}}(r,s):function(e,t){let r=e.previousElementSibling;for(;r;){if(r.matches(t))return r;r=r.previousElementSibling}}(r,s))?void 0:r.querySelector(v);n?N.setState("value",n.getAttribute(g)):G(e)}let J=()=>H(B().length-1),Q=e=>{e.preventDefault(),e.metaKey?J():e.altKey?X(1):G(1)},ee=e=>{e.preventDefault(),e.metaKey?H(0):e.altKey?X(-1):G(-1)};return c.createElement(d.WV.div,{ref:t,tabIndex:-1,...M,"cmdk-root":"",onKeyDown:e=>{var t;if(null==(t=M.onKeyDown)||t.call(M,e),!e.defaultPrevented)switch(e.key){case"n":case"j":S&&e.ctrlKey&&Q(e);break;case"ArrowDown":Q(e);break;case"p":case"k":S&&e.ctrlKey&&ee(e);break;case"ArrowUp":ee(e);break;case"Home":e.preventDefault(),H(0);break;case"End":e.preventDefault(),J();break;case"Enter":if(!e.nativeEvent.isComposing&&229!==e.keyCode){e.preventDefault();let t=Y();if(t){let e=new Event(m);t.dispatchEvent(e)}}}}},c.createElement("label",{"cmdk-label":"",htmlFor:q.inputId,id:q.labelId,style:K},i),$(e,e=>c.createElement(w.Provider,{value:N},c.createElement(b.Provider,{value:q},e))))}),k=c.forwardRef((e,t)=>{var r,n;let l=c.useId(),o=c.useRef(null),a=c.useContext(y),u=E(),i=x(e),s=null!=(n=null==(r=i.current)?void 0:r.forceMount)?n:null==a?void 0:a.forceMount;O(()=>{if(!s)return u.item(l,null==a?void 0:a.id)},[s]);let f=W(l,o,[e.value,e.children,o],e.keywords),p=C(),v=P(e=>e.value&&e.value===f.current),g=P(e=>!!s||!1===u.filter()||!e.search||e.filtered.items.get(l)>0);function h(){var e,t;b(),null==(t=(e=i.current).onSelect)||t.call(e,f.current)}function b(){p.setState("value",f.current,!0)}if(c.useEffect(()=>{let t=o.current;if(!(!t||e.disabled))return t.addEventListener(m,h),()=>t.removeEventListener(m,h)},[g,e.onSelect,e.disabled]),!g)return null;let{disabled:w,value:R,onSelect:k,forceMount:D,keywords:I,...S}=e;return c.createElement(d.WV.div,{ref:F([o,t]),...S,id:l,"cmdk-item":"",role:"option","aria-disabled":!!w,"aria-selected":!!v,"data-disabled":!!w,"data-selected":!!v,onPointerMove:w||u.disablePointerSelection?void 0:b,onClick:w?void 0:h},e.children)}),D=c.forwardRef((e,t)=>{let{heading:r,children:n,forceMount:l,...o}=e,a=c.useId(),u=c.useRef(null),i=c.useRef(null),s=c.useId(),f=E(),p=P(e=>!!l||!1===f.filter()||!e.search||e.filtered.groups.has(a));O(()=>f.group(a),[]),W(a,u,[e.value,e.heading,i]);let v=c.useMemo(()=>({id:a,forceMount:l}),[l]);return c.createElement(d.WV.div,{ref:F([u,t]),...o,"cmdk-group":"",role:"presentation",hidden:!p||void 0},r&&c.createElement("div",{ref:i,"cmdk-group-heading":"","aria-hidden":!0,id:s},r),$(e,e=>c.createElement("div",{"cmdk-group-items":"",role:"group","aria-labelledby":r?s:void 0},c.createElement(y.Provider,{value:v},e))))}),I=c.forwardRef((e,t)=>{let{alwaysRender:r,...n}=e,l=c.useRef(null),o=P(e=>!e.search);return r||o?c.createElement(d.WV.div,{ref:F([l,t]),...n,"cmdk-separator":"",role:"separator"}):null}),S=c.forwardRef((e,t)=>{let{onValueChange:r,...n}=e,l=null!=e.value,o=C(),a=P(e=>e.search),u=P(e=>e.value),i=E(),s=c.useMemo(()=>{var e;let t=null==(e=i.listInnerRef.current)?void 0:e.querySelector(`${p}[${g}="${encodeURIComponent(u)}"]`);return null==t?void 0:t.getAttribute("id")},[]);return c.useEffect(()=>{null!=e.value&&o.setState("search",e.value)},[e.value]),c.createElement(d.WV.input,{ref:t,...n,"cmdk-input":"",autoComplete:"off",autoCorrect:"off",spellCheck:!1,"aria-autocomplete":"list",role:"combobox","aria-expanded":!0,"aria-controls":i.listId,"aria-labelledby":i.labelId,"aria-activedescendant":s,id:i.inputId,type:"text",value:l?e.value:a,onChange:e=>{l||o.setState("search",e.target.value),null==r||r(e.target.value)}})}),M=c.forwardRef((e,t)=>{let{children:r,label:n="Suggestions",...l}=e,o=c.useRef(null),a=c.useRef(null),u=E();return c.useEffect(()=>{if(a.current&&o.current){let e=a.current,t=o.current,r,n=new ResizeObserver(()=>{r=requestAnimationFrame(()=>{let r=e.offsetHeight;t.style.setProperty("--cmdk-list-height",r.toFixed(1)+"px")})});return n.observe(e),()=>{cancelAnimationFrame(r),n.unobserve(e)}}},[]),c.createElement(d.WV.div,{ref:F([o,t]),...l,"cmdk-list":"",role:"listbox","aria-label":n,id:u.listId},$(e,e=>c.createElement("div",{ref:F([a,u.listInnerRef]),"cmdk-list-sizer":""},e)))}),_=c.forwardRef((e,t)=>{let{open:r,onOpenChange:n,overlayClassName:l,contentClassName:o,container:a,...u}=e;return c.createElement(i.fC,{open:r,onOpenChange:n},c.createElement(i.h_,{container:a},c.createElement(i.aV,{"cmdk-overlay":"",className:l}),c.createElement(i.VY,{"aria-label":e.label,"cmdk-dialog":"",className:o},c.createElement(R,{ref:t,...u}))))}),A=Object.assign(R,{List:M,Item:k,Input:S,Group:D,Separator:I,Dialog:_,Empty:c.forwardRef((e,t)=>P(e=>0===e.filtered.count)?c.createElement(d.WV.div,{ref:t,...e,"cmdk-empty":"",role:"presentation"}):null),Loading:c.forwardRef((e,t)=>{let{progress:r,children:n,label:l="Loading...",...o}=e;return c.createElement(d.WV.div,{ref:t,...o,"cmdk-loading":"",role:"progressbar","aria-valuenow":r,"aria-valuemin":0,"aria-valuemax":100,"aria-label":l},$(e,e=>c.createElement("div",{"aria-hidden":!0},e)))})});function x(e){let t=c.useRef(e);return O(()=>{t.current=e}),t}var O="undefined"==typeof window?c.useEffect:c.useLayoutEffect;function V(e){let t=c.useRef();return void 0===t.current&&(t.current=e()),t}function F(e){return t=>{e.forEach(e=>{"function"==typeof e?e(t):null!=e&&(e.current=t)})}}function P(e){let t=C(),r=()=>e(t.snapshot());return c.useSyncExternalStore(t.subscribe,r,r)}function W(e,t,r,n=[]){let l=c.useRef(),o=E();return O(()=>{var a;let u=(()=>{var e;for(let t of r){if("string"==typeof t)return t.trim();if("object"==typeof t&&"current"in t)return t.current?null==(e=t.current.textContent)?void 0:e.trim():l.current}})(),i=n.map(e=>e.trim());o.value(e,u,i),null==(a=t.current)||a.setAttribute(g,u),l.current=u}),l}var Z=()=>{let[e,t]=c.useState(),r=V(()=>new Map);return O(()=>{r.current.forEach(e=>e()),r.current=new Map},[e]),(e,n)=>{r.current.set(e,n),t({})}};function $({asChild:e,children:t},r){let n;return e&&c.isValidElement(t)?c.cloneElement("function"==typeof(n=t.type)?n(t.props):"render"in n?n.render(t.props):t,{ref:t.ref},r(t.props.children)):r(t)}var K={position:"absolute",width:"1px",height:"1px",padding:"0",margin:"-1px",overflow:"hidden",clip:"rect(0, 0, 0, 0)",whiteSpace:"nowrap",borderWidth:"0"}}}]);