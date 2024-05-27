"use strict";(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[2671],{52671:function(e,r,t){t.d(r,{Kb:function(){return w},fB:function(){return C}});var n=t(36164),a=t(3546),s=t(70652),i=t.n(s),o=t(11978),l=t(84381),d=t(94909),c=t(5493),u=t(2578),f=t(23782),m=t(18500),x=t(74248),p=t(73460),h=t(31458),g=t(98150),v=t(81565),b=t(82394),j=t(62066);let y=f.Ry({displayName:f.Z_().trim(),accessToken:f.Z_()}),N=y.extend({accessToken:f.Z_().optional()});function w(e){let{isNew:r,form:t,onSubmit:s,onDelete:i,cancleable:l=!0,deletable:c}=e,f=(0,j.Y)(),y=(0,o.useRouter)(),[N,w]=a.useState(!1),[C,R]=a.useState(!1),{isSubmitting:_,dirtyFields:z}=t.formState,D=!(0,d.Z)(z),G=async e=>{if(e.preventDefault(),i){R(!0);try{await i()}catch(e){u.A.error("Failed to delete GitHub repository provider")}finally{R(!1)}}},I=a.useMemo(()=>{switch(f){case m.vW.Github:return"e.g. GitHub";case m.vW.Gitlab:return"e.g. GitLab";default:return""}},[f]),A=a.useMemo(()=>{if(!r)return Array(36).fill("*").join("");switch(f){case m.vW.Github:return"e.g. github_pat_1ABCD1234ABCD1234ABCD1234ABCD1234ABCD1234";case m.vW.Gitlab:return"e.g. glpat_1ABCD1234ABCD1234ABCD1234ABCD1234";default:return""}},[f,r]);return(0,n.jsx)(g.l0,{...t,children:(0,n.jsx)("div",{className:"grid gap-2",children:(0,n.jsxs)("form",{className:"grid gap-6",onSubmit:t.handleSubmit(s),children:[(0,n.jsx)(g.Wi,{control:t.control,name:"displayName",render:e=>{let{field:r}=e;return(0,n.jsxs)(g.xJ,{children:[(0,n.jsx)(g.lX,{required:!0,children:"Display name"}),(0,n.jsx)(g.pf,{children:"A display name to help identifying different providers."}),(0,n.jsx)(g.NI,{children:(0,n.jsx)(b.I,{placeholder:I,autoCapitalize:"none",autoCorrect:"off",autoComplete:"off",...r})}),(0,n.jsx)(g.zG,{})]})}}),(0,n.jsx)(g.Wi,{control:t.control,name:"accessToken",render:e=>{let{field:t}=e;return(0,n.jsxs)(g.xJ,{children:[(0,n.jsx)(g.lX,{required:!0,children:"Personal Access Token"}),(0,n.jsx)(g.pf,{children:(0,n.jsx)(k,{})}),(0,n.jsx)(g.NI,{children:(0,n.jsx)(b.I,{placeholder:A,className:(0,x.cn)({"placeholder:translate-y-[10%] !placeholder-foreground":!r}),autoCapitalize:"none",autoCorrect:"off",autoComplete:"off",...t})}),(0,n.jsx)(g.zG,{})]})}}),(0,n.jsxs)("div",{className:"flex items-center justify-between",children:[(0,n.jsx)("div",{children:(0,n.jsx)(g.zG,{})}),(0,n.jsxs)("div",{className:"flex gap-2",children:[l&&(0,n.jsx)(h.z,{type:"button",variant:"ghost",disabled:_,onClick:()=>y.back(),children:"Cancel"}),c&&(0,n.jsxs)(p.aR,{open:N,onOpenChange:w,children:[(0,n.jsx)(p.vW,{asChild:!0,children:(0,n.jsx)(h.z,{type:"button",variant:"hover-destructive",children:"Delete"})}),(0,n.jsxs)(p._T,{children:[(0,n.jsxs)(p.fY,{children:[(0,n.jsx)(p.f$,{children:"Are you absolutely sure?"}),(0,n.jsx)(p.yT,{children:"This will delete the provider and remove any repositories that have already been added to the provider."})]}),(0,n.jsxs)(p.xo,{children:[(0,n.jsx)(p.le,{children:"Cancel"}),(0,n.jsxs)(p.OL,{className:(0,h.d)({variant:"destructive"}),onClick:G,disabled:C,children:[C&&(0,n.jsx)(v.IconSpinner,{className:"mr-2"}),"Yes, delete it"]})]})]})]}),(0,n.jsxs)(h.z,{type:"submit",disabled:_||!r&&!D,children:[_&&(0,n.jsx)(v.IconSpinner,{className:"mr-2"}),r?"Create":"Update"]})]})]})]})})})}function C(e,r){let t=e?y:N;return(0,c.cI)({resolver:(0,l.F)(t),defaultValues:r})}function k(){let e=(0,j.Y)();return e===m.vW.Github?(0,n.jsxs)(n.Fragment,{children:[(0,n.jsxs)("div",{children:["Create a dedicated service user and generate a"," ",(0,n.jsx)(R,{href:"https://github.com/settings/personal-access-tokens/new",children:"fine-grained personal access"})," ","token with the member role for the organization or all projects to be managed."]}),(0,n.jsx)("div",{className:"my-2 ml-3",children:"• Contents (Read-only)"})]}):e===m.vW.Gitlab?(0,n.jsxs)(n.Fragment,{children:[(0,n.jsxs)("div",{children:["Create a dedicated service user and generate a"," ",(0,n.jsx)(R,{href:"https://gitlab.com/-/user_settings/personal_access_tokens",children:"personal access token"})," ","with the maintainer role and at least following permissions for the group or projects to be managed. You can generate a project access token for managing a single project, or generate a group access token to manage all projects within the group."]}),(0,n.jsx)("div",{className:"my-2 ml-3",children:"• api"}),(0,n.jsx)("div",{className:"my-2 ml-3",children:"• read repository"})]}):null}function R(e){let{href:r,children:t}=e;return(0,n.jsxs)(i(),{className:"inline-flex cursor-pointer flex-row items-center underline",href:r,target:"_blank",children:[t,(0,n.jsx)(v.IconExternalLink,{className:"ml-1"})]})}},62066:function(e,r,t){t.d(r,{Y:function(){return o}});var n=t(11978),a=t(54549),s=t(18500);let i=[{name:"github",enum:s.vW.Github,meta:{domain:"github.com",displayName:"GitHub"}},{name:"gitlab",enum:s.vW.Gitlab,meta:{domain:"gitlab.com",displayName:"GitLab"}}];function o(){let e=(0,n.useParams)(),r=(0,a.Z)(i,r=>{var t;return r.enum.toLocaleLowerCase()===(null===(t=e.kind)||void 0===t?void 0:t.toLocaleLowerCase())}),t=r>-1?i[r].enum:i[0].enum;return t}},73460:function(e,r,t){t.d(r,{OL:function(){return g},_T:function(){return f},aR:function(){return l},f$:function(){return p},fY:function(){return m},le:function(){return v},vW:function(){return d},xo:function(){return x},yT:function(){return h}});var n=t(36164),a=t(3546),s=t(34604),i=t(74248),o=t(31458);let l=s.fC,d=s.xz,c=e=>{let{className:r,children:t,...a}=e;return(0,n.jsx)(s.h_,{className:(0,i.cn)(r),...a,children:(0,n.jsx)("div",{className:"fixed inset-0 z-50 flex items-end justify-center sm:items-center",children:t})})};c.displayName=s.h_.displayName;let u=a.forwardRef((e,r)=>{let{className:t,children:a,...o}=e;return(0,n.jsx)(s.aV,{className:(0,i.cn)("fixed inset-0 z-50 bg-background/80 backdrop-blur-sm transition-opacity animate-in fade-in",t),...o,ref:r})});u.displayName=s.aV.displayName;let f=a.forwardRef((e,r)=>{let{className:t,...a}=e;return(0,n.jsxs)(c,{children:[(0,n.jsx)(u,{}),(0,n.jsx)(s.VY,{ref:r,className:(0,i.cn)("fixed z-50 grid w-full max-w-lg scale-100 gap-4 border bg-background p-6 opacity-100 shadow-lg animate-in fade-in-90 slide-in-from-bottom-10 sm:rounded-lg sm:zoom-in-90 sm:slide-in-from-bottom-0 md:w-full",t),...a})]})});f.displayName=s.VY.displayName;let m=e=>{let{className:r,...t}=e;return(0,n.jsx)("div",{className:(0,i.cn)("flex flex-col space-y-2 text-center sm:text-left",r),...t})};m.displayName="AlertDialogHeader";let x=e=>{let{className:r,...t}=e;return(0,n.jsx)("div",{className:(0,i.cn)("flex flex-col-reverse sm:flex-row sm:justify-end sm:space-x-2",r),...t})};x.displayName="AlertDialogFooter";let p=a.forwardRef((e,r)=>{let{className:t,...a}=e;return(0,n.jsx)(s.Dx,{ref:r,className:(0,i.cn)("text-lg font-semibold",t),...a})});p.displayName=s.Dx.displayName;let h=a.forwardRef((e,r)=>{let{className:t,...a}=e;return(0,n.jsx)(s.dk,{ref:r,className:(0,i.cn)("text-sm text-muted-foreground",t),...a})});h.displayName=s.dk.displayName;let g=a.forwardRef((e,r)=>{let{className:t,...a}=e;return(0,n.jsx)(s.aU,{ref:r,className:(0,i.cn)((0,o.d)(),t),...a})});g.displayName=s.aU.displayName;let v=a.forwardRef((e,r)=>{let{className:t,...a}=e;return(0,n.jsx)(s.$j,{ref:r,className:(0,i.cn)((0,o.d)({variant:"outline"}),"mt-2 sm:mt-0",t),...a})});v.displayName=s.$j.displayName},31458:function(e,r,t){t.d(r,{d:function(){return l},z:function(){return d}});var n=t(36164),a=t(3546),s=t(74047),i=t(14375),o=t(74248);let l=(0,i.j)("inline-flex items-center justify-center rounded-md text-sm font-medium shadow ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",{variants:{variant:{default:"bg-primary text-primary-foreground shadow-md hover:bg-primary/90",destructive:"bg-destructive text-destructive-foreground hover:bg-destructive/90","hover-destructive":"shadow-none hover:bg-destructive/90 hover:text-destructive-foreground",outline:"border border-input hover:bg-accent hover:text-accent-foreground",secondary:"bg-secondary text-secondary-foreground hover:bg-secondary/80",ghost:"shadow-none hover:bg-accent hover:text-accent-foreground",link:"text-primary underline-offset-4 shadow-none hover:underline"},size:{default:"h-8 px-4 py-2",sm:"h-8 rounded-md px-3",lg:"h-11 rounded-md px-8",icon:"h-8 w-8 p-0"}},defaultVariants:{variant:"default",size:"default"}}),d=a.forwardRef((e,r)=>{let{className:t,variant:a,size:i,asChild:d=!1,...c}=e,u=d?s.g7:"button";return(0,n.jsx)(u,{className:(0,o.cn)(l({variant:a,size:i,className:t})),ref:r,...c})});d.displayName="Button"},98150:function(e,r,t){t.d(r,{NI:function(){return h},Wi:function(){return u},l0:function(){return d},lX:function(){return p},pf:function(){return g},xJ:function(){return x},zG:function(){return v}});var n=t(36164),a=t(3546),s=t(74047),i=t(5493),o=t(74248),l=t(5266);let d=i.RV,c=a.createContext({}),u=e=>{let{...r}=e;return(0,n.jsx)(c.Provider,{value:{name:r.name},children:(0,n.jsx)(i.Qr,{...r})})},f=()=>{let e=a.useContext(c),r=a.useContext(m),{getFieldState:t,formState:n}=(0,i.Gc)(),s=e.name||"root",o=t(s,n);if(!n)throw Error("useFormField should be used within <Form>");let{id:l}=r;return{id:l,name:s,formItemId:"".concat(l,"-form-item"),formDescriptionId:"".concat(l,"-form-item-description"),formMessageId:"".concat(l,"-form-item-message"),...o}},m=a.createContext({}),x=a.forwardRef((e,r)=>{let{className:t,...s}=e,i=a.useId();return(0,n.jsx)(m.Provider,{value:{id:i},children:(0,n.jsx)("div",{ref:r,className:(0,o.cn)("space-y-2",t),...s})})});x.displayName="FormItem";let p=a.forwardRef((e,r)=>{let{className:t,required:a,...s}=e,{error:i,formItemId:d}=f();return(0,n.jsx)(l._,{ref:r,className:(0,o.cn)(i&&"text-destructive",a&&'after:ml-0.5 after:text-destructive after:content-["*"]',t),htmlFor:d,...s})});p.displayName="FormLabel";let h=a.forwardRef((e,r)=>{let{...t}=e,{error:a,formItemId:i,formDescriptionId:o,formMessageId:l}=f();return(0,n.jsx)(s.g7,{ref:r,id:i,"aria-describedby":a?"".concat(o," ").concat(l):"".concat(o),"aria-invalid":!!a,...t})});h.displayName="FormControl";let g=a.forwardRef((e,r)=>{let{className:t,...a}=e,{formDescriptionId:s}=f();return(0,n.jsx)("div",{ref:r,id:s,className:(0,o.cn)("text-sm text-muted-foreground",t),...a})});g.displayName="FormDescription";let v=a.forwardRef((e,r)=>{let{className:t,children:a,...s}=e,{error:i,formMessageId:l}=f(),d=i?String(null==i?void 0:i.message):a;return d?(0,n.jsx)("p",{ref:r,id:l,className:(0,o.cn)("text-sm font-medium text-destructive",t),...s,children:d}):null});v.displayName="FormMessage"},82394:function(e,r,t){t.d(r,{I:function(){return i}});var n=t(36164),a=t(3546),s=t(74248);let i=a.forwardRef((e,r)=>{let{className:t,type:a,...i}=e;return(0,n.jsx)("input",{type:a,className:(0,s.cn)("flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",t),ref:r,...i})});i.displayName="Input"},5266:function(e,r,t){t.d(r,{_:function(){return d}});var n=t(36164),a=t(3546),s=t(90893),i=t(14375),o=t(74248);let l=(0,i.j)("text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"),d=a.forwardRef((e,r)=>{let{className:t,...a}=e;return(0,n.jsx)(s.f,{ref:r,className:(0,o.cn)(l(),t),...a})});d.displayName=s.f.displayName}}]);