(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[7844],{47844:function(e,t,n){"use strict";let r;n.r(t),n.d(t,{default:function(){return Y}});var i=n(36164),o=n(3546),l=n(30352),a=n(88105),s=n(74630),d=n(28242),c=n(28312),u=n(11978),m=n(16784),f=n(1544),h=n(85312);function p(e,t){let n=e.doc,{span:r,utf16_column_range:i}=t;try{let e=n.line(r.start.row+1),t=e.from+i.start,o=e.from+i.end;return{start:t,end:o}}catch(e){return null}}let v=a.p.mark({class:"cm-tag-mark"}),g=a.tk.baseTheme({".cm-tag-mark":{border:"1px solid hsla(var(--tag-blue-border))",padding:"0px 4px",borderRadius:"4px",backgroundColor:"hsla(var(--tag-blue-bg))",color:"hsla(var(--tag-blue-text)) !important"},".cm-tag-mark > span":{color:"hsla(var(--tag-blue-text)) !important"}});function x(e){let t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:[],n=e.state.doc,r=n.length;if(!r)return a.p.none;let i=[];for(let n of t){let t=p(e.state,n);try{t&&t.start<=r&&t.end<=r&&i.push(v.range(t.start,t.end))}catch(e){}}return i.length?a.p.set(i):a.p.none}let b=e=>{let t=a.lg.fromClass(class{update(t){(t.docChanged||t.viewportChanged)&&(this.marks=x(t.view,e))}constructor(t){this.marks=x(t,e)}},{decorations:e=>e.marks});return[t,g]};var w=n(94559);let y=a.p.mark({class:"cm-range-highlight"}),k=a.tk.baseTheme({".cm-range-highlight":{backgroundColor:"hsl(var(--selection))"}});function _(e,t){let n;let r=e.selection.ranges;e:for(let i of r)for(let r of t){let t=p(e,r);if(!t)continue;let o=t.start-r.name_range.start;if(i.from>=t.start&&i.to<=t.end){n={from:r.range.start+o,to:r.range.end+o};break e}}return n?a.p.set([y.range(n.from,n.to)]):a.p.none}let j=w.Py.define(),C=e=>{let t=a.lg.fromClass(class{update(t){if(t.selectionSet)this.triggerType="cursor",this.highlight=_(t.view.state,e);else if("cursor"!==this.triggerType)for(let e of t.transactions)for(let t of e.effects)t.is(j)&&t.value&&(this.highlight=t.value,this.triggerType="hover")}handleMouseListener(t){if(-1!==this.timeout&&clearTimeout(this.timeout),!this.highlight.size){let n=setTimeout(()=>{let n=this.view.posAtCoords({x:t.clientX,y:t.clientY});if(null!==n){let t=function(e,t){let n;for(let r of t)if(e>=r.name_range.start&&e<=r.name_range.end){n={from:r.range.start,to:r.range.end};break}return n?a.p.set([y.range(n.from,n.to)]):a.p.none}(n,e);if(t.size)this.triggerType="hover";else if("cursor"===this.triggerType)return;this.view.dispatch({effects:j.of(t)})}},100);this.timeout=n}}destroy(){}constructor(t){this.view=t,this.highlight=_(t.state,e),this.timeout=-1,this.triggerType="hover"}},{decorations:e=>e.highlight});return[t,k]},N=a.tk.baseTheme({".cm-tooltip":{border:"none !important"},".cm-tooltip-cursor":{backgroundColor:"hsl(var(--popover))",color:"hsl(var(--popover-foreground))",border:"none !important",padding:"2px 7px",borderRadius:"4px"}}),L=e=>[(0,a.bF)((t,n,r)=>{for(let r of e){let e=p(t.state,r);if(e&&n>=e.start&&n<=e.end)return{pos:n,above:!0,create(e){let t=document.createElement("div");return t.className="cm-tooltip-cursor",t.textContent="".concat(r.syntax_type_name),{dom:t,offset:{x:-20,y:4}}}}}return null}),N];var M=n(27064),E=n(87279),T=n(42891),S=n.n(T),R=n(23342),U=n(31458),X=n(62202),z=n(81565);let A=e=>{let{className:t,text:n,language:r,path:o,lineFrom:l,lineTo:a,gitUrl:s,...d}=e,c=e=>{M.u.emit("code_browser_quick_action",{action:e,code:n,language:r,path:o,lineFrom:l,lineTo:a,gitUrl:s})};return(0,i.jsxs)("div",{className:(0,f.cn)("mt-2 flex items-center gap-2 rounded-md border bg-background px-2 py-1",t),...d,children:[(0,i.jsx)(S(),{src:R.Z,width:32,alt:"logo"}),(0,i.jsx)(U.z,{size:"sm",variant:"outline",onClick:e=>c("explain"),children:"Explain"}),(0,i.jsxs)(X.h_,{modal:!1,children:[(0,i.jsx)(X.$F,{asChild:!0,children:(0,i.jsxs)(U.z,{size:"sm",variant:"outline",children:["Generate",(0,i.jsx)(z.IconChevronUpDown,{className:"ml-1"})]})}),(0,i.jsxs)(X.AW,{align:"start",children:[(0,i.jsx)(X.Xi,{className:"cursor-pointer",onSelect:()=>c("generate_unittest"),children:"Unit Test"}),(0,i.jsx)(X.Xi,{className:"cursor-pointer",onSelect:()=>c("generate_doc"),children:"Documentation"})]})]})]})},Z=new class extends a.SJ{constructor(...e){super(...e),this.elementClass="cm-selectedLineGutter"}},D=w.Py.define(),P=w.QQ.define({create:()=>null,update(e,t){for(let n of t.effects){if(n.is(D))return n.value;if(n.is(J))return(null==e?void 0:e.line)||(e={line:n.value}),{...e,endLine:n.value}}return e},provide:e=>[a.tk.decorations.compute([e],t=>{var n;let r=t.field(e);if(!r)return a.p.none;let i=null!==(n=r.endLine)&&void 0!==n?n:r.line,o=Math.min(r.line,i),l=Math.min(t.doc.lines,o===i?r.line:i),s=new w.f_;for(let e=o;e<=l;e++){let n=t.doc.line(e).from;s.add(n,n,a.p.line({class:"cm-selectedLine"}))}return s.finish()}),a.v7.compute([e],t=>{var n;let r=[],i=t.field(e);if(!i)return w.Xs.empty;let o=null!==(n=i.endLine)&&void 0!==n?n:i.line,l=Math.min(i.line,o),a=Math.min(t.doc.lines,l===o?i.line:o);for(let e=l;e<=a;e++){let n=t.doc.line(e).from;r.push(Z.range(n))}return w.Xs.of(r)})]}),F=e=>{let{isMulti:t}=e;return(0,i.jsxs)(X.h_,{modal:!1,children:[(0,i.jsx)(X.$F,{asChild:!0,children:(0,i.jsx)(U.z,{className:"ml-1 h-5",size:"icon",variant:"secondary",children:(0,i.jsx)(z.IconMore,{})})}),(0,i.jsxs)(X.AW,{align:"start",children:[(0,i.jsx)(X.Xi,{className:"cursor-pointer",onSelect:()=>{M.u.emit("line_menu_action",{action:"copy_line"})},children:t?"Copy lines":"Copy line"}),(0,i.jsx)(X.Xi,{className:"cursor-pointer",onSelect:e=>{M.u.emit("line_menu_action",{action:"copy_permalink"})},children:"Copy permalink"})]})]})},H=new class extends a.SJ{toDOM(){let e=document.createElement("div"),t=E.createRoot(e);return t.render((0,i.jsx)(F,{isMulti:!1})),e}},Q=new class extends a.SJ{toDOM(){let e=document.createElement("div"),t=E.createRoot(e);return t.render((0,i.jsx)(F,{isMulti:!0})),e}},B=a.tk.theme({".cm-lineMenuGutter":{width:"40px"},".cm-lineNumbers":{userSelect:"none",cursor:"pointer",color:"var(--line-number-color)","& .cm-gutterElement:hover":{textDecoration:"underline"}}});function I(e,t){let n=G(t,e.state.doc);return e.dispatch({effects:D.of(n?t:null)}),t}let J=w.Py.define(),O=e=>{let{onSelectLine:t}=e;return[B,P,(0,a.v5)({class:"cm-lineMenuGutter",markers:e=>(function(e){let t=e.state.field(P);if(!G(t,e.state.doc))return w.Xs.empty;if(null==t?void 0:t.line){let n=!!t.endLine&&t.line!==t.endLine,r=t.endLine?Math.min(t.line,t.endLine):t.line,i=e.state.doc.line(r).from;return w.Xs.empty.update({add:[n?Q.range(i):H.range(i)]})}return w.Xs.empty})(e),initialSpacer:()=>H,domEventHandlers:{mousedown(e,t,n){let r=e.state.doc.lineAt(t.from),i=r.number;return e.dispatch({effects:n.shiftKey?J.of(i):D.of({line:i})}),!0}}}),(0,a.Eu)({domEventHandlers:{mousedown(e,n,r){let i=e.state.doc.lineAt(n.from).number;return e.dispatch({effects:r.shiftKey?J.of(i):D.of({line:i})}),null==t||t(function(e){if(!e)return;let{line:t,endLine:n}=e;return t&&n?t===n?{line:t}:{line:Math.min(t,n),endLine:Math.max(t,n)}:t?{line:t}:void 0}(e.state.field(P))),!1}}})]};function G(e,t){if(!t)return!1;let{lines:n}=t;return(null==e||!e.line||!(e.line>n))&&(null==e||!e.endLine||!(e.endLine>n))}var V=n(53585),$=n(48048);n(85223);var W=n(63484),Y=e=>{var t,n,p;let{value:v,language:g}=e,{theme:x}=(0,d.F)(),y=o.useMemo(()=>[],[]),{copyToClipboard:k}=(0,c.m)({}),[_,j]=function(){let e=(0,u.useParams)(),[t,n]=o.useState(""),r=(0,m.d)(t),i=o.useCallback(e=>{window.location.hash=e},[]),l=()=>{let e=window.location.hash;r.current!==e&&n(e)};return o.useEffect(()=>(window.addEventListener("hashchange",l),()=>{window.removeEventListener("hashchange",l)}),[]),o.useEffect(l,[e]),[t,i]}(),N=null===(t=(0,$.kQ)(_))||void 0===t?void 0:t.start,T=null===(n=(0,$.kQ)(_))||void 0===n?void 0:n.end,[S,R]=o.useState(null),{isChatEnabled:U,activePath:X,activeEntryInfo:z,activeRepo:Z,activeRepoRef:D}=o.useContext(V.SourceCodeBrowserContext),{basename:P}=z,F=null!==(p=null==Z?void 0:Z.gitUrl)&&void 0!==p?p:"",H=o.useMemo(()=>{let e=[O({onSelectLine:e=>{if(!e){j("");return}j((0,f.nO)({start:e.line,end:e.endLine}))}}),(0,l.mi)({markerDOM(e){let t=document.createElement("div");return t.style.cursor="pointer",e?t.innerHTML='<svg aria-hidden="true" focusable="false" role="img" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" style="display: inline-block; user-select: none; vertical-align: text-bottom; overflow: visible;"><path d="M12.78 5.22a.749.749 0 0 1 0 1.06l-4.25 4.25a.749.749 0 0 1-1.06 0L3.22 6.28a.749.749 0 1 1 1.06-1.06L8 8.939l3.72-3.719a.749.749 0 0 1 1.06 0Z"></path></svg>':t.innerHTML='<svg aria-hidden="true" focusable="false" role="img" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" style="display: inline-block; user-select: none; vertical-align: text-bottom; overflow: visible;"><path d="M6.22 3.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L9.94 8 6.22 4.28a.75.75 0 0 1 0-1.06Z"></path></svg>',t}}),(0,a.Uw)()];if(U&&X&&P){var t;e.push((t={language:g,path:P,gitUrl:F},w.QQ.define({create:()=>null,update(e,n){if(n.newSelection.main.empty)return clearTimeout(r),null;if(n.selection){if(function(e){let t=!!e.selection&&!e.selection.main.empty;return t&&e.isUserEvent("select")}(n)){let e=function(e,t){let{selection:n}=e,o=e.doc.lineAt(n.main.from),l=e.doc.lineAt(n.main.to),a=o.number!==l.number,s=a?l.from:n.main.from,d=e.doc.sliceString(e.selection.main.from,e.selection.main.to)||"";return{pos:s,above:!1,strictSide:!0,arrow:!1,create(){let e=document.createElement("div");e.style.background="transparent",e.style.border="none";let n=E.createRoot(e);return e.onclick=e=>e.stopImmediatePropagation(),r&&clearTimeout(r),r=window.setTimeout(()=>{n.render((0,i.jsx)(A,{text:d,language:null==t?void 0:t.language,lineFrom:o.number,lineTo:l.number,path:null==t?void 0:t.path,gitUrl:null==t?void 0:t.gitUrl}))},1e3),{dom:e}}}}(n.state,t);return e}return clearTimeout(r),null}return e},provide:e=>a.hJ.compute([e],t=>t.field(e))})))}return v&&y&&e.push(b(y),L(y),C(y)),e},[v,y,g]);return o.useEffect(()=>{let e=e=>{if("number"==typeof N){if("copy_permalink"===e.action){var t,n,r;let e=(0,$.I)(Z,null!==(n=null==D?void 0:null===(t=D.ref)||void 0===t?void 0:t.commit)&&void 0!==n?n:null==D?void 0:D.name,null!==(r=z.basename)&&void 0!==r?r:"",(0,$.BX)(z.viewMode)),i=new URL("".concat(window.location.origin,"/files/").concat(e));(0,$.p4)(window.location.hash)&&(i.hash=window.location.hash);let o=z.basename?(0,W.U$)(z.basename)[0]:void 0;"markdown"===o&&i.searchParams.set("plain","1"),k(i.toString());return}if("copy_line"===e.action){let e,t;if(!S)return;let n=S.state.doc.line(N);if(T&&(t=S.state.doc.line(T)),n&&t&&n.number<=t.number){let r=n.from,i=t.to;e=S.state.doc.slice(r,i).toString()}else n&&(e=n.text);e&&k(e)}}};return M.u.on("line_menu_action",e),()=>{M.u.off("line_menu_action",e)}},[v,N,T,S]),o.useEffect(()=>{if(!(0,s.Z)(N)&&S&&v)try{var e,t,n,r;let i=null==S?void 0:null===(t=S.state)||void 0===t?void 0:null===(e=t.doc)||void 0===e?void 0:e.line(N),o=(0,s.Z)(T)?null:null==S?void 0:null===(r=S.state)||void 0===r?void 0:null===(n=r.doc)||void 0===n?void 0:n.line(T);if(i){let e=i.number,t=null==o?void 0:o.number;if(I(S,{line:e,endLine:t}),function(e,t){let n=arguments.length>2&&void 0!==arguments[2]?arguments[2]:0,r=e.domAtPos(t).node,i=3===r.nodeType?r.parentElement:r;if(i){let e=i.getBoundingClientRect(),t=window.innerHeight||document.documentElement.clientHeight;return e.top>=n&&e.bottom<=t}return!1}(S,i.from,90))return;S.dispatch({effects:a.tk.scrollIntoView(i.from,{y:"start",yMargin:200})})}}catch(e){}return()=>{S&&I(S,null)}},[v,N,S]),(0,i.jsx)(h.Z,{value:v,theme:x,language:g,readonly:!0,extensions:H,viewDidUpdate:e=>R(e)})}},62202:function(e,t,n){"use strict";n.d(t,{$F:function(){return s},AW:function(){return c},Ju:function(){return m},VD:function(){return f},Xi:function(){return u},h_:function(){return a}});var r=n(36164),i=n(3546),o=n(19148),l=n(1544);let a=o.fC,s=o.xz;o.ZA,o.Uv,o.Tr,o.Ee;let d=i.forwardRef((e,t)=>{let{className:n,...i}=e;return(0,r.jsx)(o.tu,{ref:t,className:(0,l.cn)("z-50 min-w-[8rem] overflow-hidden rounded-md border bg-popover p-1 text-popover-foreground shadow-md animate-in data-[side=bottom]:slide-in-from-top-1 data-[side=left]:slide-in-from-right-1 data-[side=right]:slide-in-from-left-1 data-[side=top]:slide-in-from-bottom-1",n),...i})});d.displayName=o.tu.displayName;let c=i.forwardRef((e,t)=>{let{className:n,sideOffset:i=4,...a}=e;return(0,r.jsx)(o.Uv,{children:(0,r.jsx)(o.VY,{ref:t,sideOffset:i,className:(0,l.cn)("z-50 min-w-[8rem] overflow-hidden rounded-md border bg-popover p-1 text-popover-foreground shadow animate-in data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",n),...a})})});c.displayName=o.VY.displayName;let u=i.forwardRef((e,t)=>{let{className:n,inset:i,...a}=e;return(0,r.jsx)(o.ck,{ref:t,className:(0,l.cn)("relative flex cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none transition-colors focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50",i&&"pl-8",n),...a})});u.displayName=o.ck.displayName;let m=i.forwardRef((e,t)=>{let{className:n,inset:i,...a}=e;return(0,r.jsx)(o.__,{ref:t,className:(0,l.cn)("px-2 py-1.5 text-sm font-semibold",i&&"pl-8",n),...a})});m.displayName=o.__.displayName;let f=i.forwardRef((e,t)=>{let{className:n,...i}=e;return(0,r.jsx)(o.Z0,{ref:t,className:(0,l.cn)("-mx-1 my-1 h-px bg-muted",n),...i})});f.displayName=o.Z0.displayName},85223:function(){}}]);