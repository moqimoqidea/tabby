"use strict";(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[1386],{37061:function(r,e,t){t.d(e,{W:function(){return o}});function o(){for(var r,e,t=0,o="";t<arguments.length;)(r=arguments[t++])&&(e=function r(e){var t,o,n="";if("string"==typeof e||"number"==typeof e)n+=e;else if("object"==typeof e){if(Array.isArray(e))for(t=0;t<e.length;t++)e[t]&&(o=r(e[t]))&&(n&&(n+=" "),n+=o);else for(t in e)e[t]&&(n&&(n+=" "),n+=t)}return n}(r))&&(o&&(o+=" "),o+=e);return o}},75623:function(r,e,t){/**
 * @license React
 * react-jsx-runtime.production.min.js
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var o=t(3546),n=Symbol.for("react.element"),i=Symbol.for("react.fragment"),a=Object.prototype.hasOwnProperty,l=o.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner,s={key:!0,ref:!0,__self:!0,__source:!0};function c(r,e,t){var o,i={},c=null,u=null;for(o in void 0!==t&&(c=""+t),void 0!==e.key&&(c=""+e.key),void 0!==e.ref&&(u=e.ref),e)a.call(e,o)&&!s.hasOwnProperty(o)&&(i[o]=e[o]);if(r&&r.defaultProps)for(o in e=r.defaultProps)void 0===i[o]&&(i[o]=e[o]);return{$$typeof:n,type:r,key:c,ref:u,props:i,_owner:l.current}}e.Fragment=i,e.jsx=c,e.jsxs=c},36164:function(r,e,t){r.exports=t(75623)},22441:function(r,e,t){t.d(e,{Z:function(){return l}});var o=t(77725),n=function(r,e){for(var t=r.length;t--;)if((0,o.Z)(r[t][0],e))return t;return -1},i=Array.prototype.splice;function a(r){var e=-1,t=null==r?0:r.length;for(this.clear();++e<t;){var o=r[e];this.set(o[0],o[1])}}a.prototype.clear=function(){this.__data__=[],this.size=0},a.prototype.delete=function(r){var e=this.__data__,t=n(e,r);return!(t<0)&&(t==e.length-1?e.pop():i.call(e,t,1),--this.size,!0)},a.prototype.get=function(r){var e=this.__data__,t=n(e,r);return t<0?void 0:e[t][1]},a.prototype.has=function(r){return n(this.__data__,r)>-1},a.prototype.set=function(r,e){var t=this.__data__,o=n(t,r);return o<0?(++this.size,t.push([r,e])):t[o][1]=e,this};var l=a},98512:function(r,e,t){var o=t(47404),n=t(48717),i=(0,o.Z)(n.Z,"Map");e.Z=i},26541:function(r,e,t){t.d(e,{Z:function(){return f}});var o=(0,t(47404).Z)(Object,"create"),n=Object.prototype.hasOwnProperty,i=Object.prototype.hasOwnProperty;function a(r){var e=-1,t=null==r?0:r.length;for(this.clear();++e<t;){var o=r[e];this.set(o[0],o[1])}}a.prototype.clear=function(){this.__data__=o?o(null):{},this.size=0},a.prototype.delete=function(r){var e=this.has(r)&&delete this.__data__[r];return this.size-=e?1:0,e},a.prototype.get=function(r){var e=this.__data__;if(o){var t=e[r];return"__lodash_hash_undefined__"===t?void 0:t}return n.call(e,r)?e[r]:void 0},a.prototype.has=function(r){var e=this.__data__;return o?void 0!==e[r]:i.call(e,r)},a.prototype.set=function(r,e){var t=this.__data__;return this.size+=this.has(r)?0:1,t[r]=o&&void 0===e?"__lodash_hash_undefined__":e,this};var l=t(22441),s=t(98512),c=function(r){var e=typeof r;return"string"==e||"number"==e||"symbol"==e||"boolean"==e?"__proto__"!==r:null===r},u=function(r,e){var t=r.__data__;return c(e)?t["string"==typeof e?"string":"hash"]:t.map};function d(r){var e=-1,t=null==r?0:r.length;for(this.clear();++e<t;){var o=r[e];this.set(o[0],o[1])}}d.prototype.clear=function(){this.size=0,this.__data__={hash:new a,map:new(s.Z||l.Z),string:new a}},d.prototype.delete=function(r){var e=u(this,r).delete(r);return this.size-=e?1:0,e},d.prototype.get=function(r){return u(this,r).get(r)},d.prototype.has=function(r){return u(this,r).has(r)},d.prototype.set=function(r,e){var t=u(this,r),o=t.size;return t.set(r,e),this.size+=t.size==o?0:1,this};var f=d},57390:function(r,e,t){var o=t(47404),n=t(48717),i=(0,o.Z)(n.Z,"Set");e.Z=i},73576:function(r,e,t){t.d(e,{Z:function(){return i}});var o=t(26541);function n(r){var e=-1,t=null==r?0:r.length;for(this.__data__=new o.Z;++e<t;)this.add(r[e])}n.prototype.add=n.prototype.push=function(r){return this.__data__.set(r,"__lodash_hash_undefined__"),this},n.prototype.has=function(r){return this.__data__.has(r)};var i=n},7600:function(r,e,t){var o=t(48717).Z.Symbol;e.Z=o},96703:function(r,e){e.Z=function(r,e,t,o){for(var n=r.length,i=t+(o?1:-1);o?i--:++i<n;)if(e(r[i],i,r))return i;return -1}},17996:function(r,e,t){t.d(e,{Z:function(){return d}});var o=t(7600),n=Object.prototype,i=n.hasOwnProperty,a=n.toString,l=o.Z?o.Z.toStringTag:void 0,s=function(r){var e=i.call(r,l),t=r[l];try{r[l]=void 0;var o=!0}catch(r){}var n=a.call(r);return o&&(e?r[l]=t:delete r[l]),n},c=Object.prototype.toString,u=o.Z?o.Z.toStringTag:void 0,d=function(r){return null==r?void 0===r?"[object Undefined]":"[object Null]":u&&u in Object(r)?s(r):c.call(r)}},62792:function(r,e,t){t.d(e,{Z:function(){return a}});var o=t(96703),n=function(r){return r!=r},i=function(r,e,t){for(var o=t-1,n=r.length;++o<n;)if(r[o]===e)return o;return -1},a=function(r,e,t){return e==e?i(r,e,t):(0,o.Z)(r,n,t)}},19108:function(r,e,t){t.d(e,{Z:function(){return f}});var o=t(73576),n=t(62792),i=function(r,e){return!!(null==r?0:r.length)&&(0,n.Z)(r,e,0)>-1},a=function(r,e,t){for(var o=-1,n=null==r?0:r.length;++o<n;)if(t(e,r[o]))return!0;return!1},l=t(14658),s=t(57390),c=t(71480),u=t(59410),d=s.Z&&1/(0,u.Z)(new s.Z([,-0]))[1]==1/0?function(r){return new s.Z(r)}:c.Z,f=function(r,e,t){var n=-1,s=i,c=r.length,f=!0,p=[],b=p;if(t)f=!1,s=a;else if(c>=200){var g=e?null:d(r);if(g)return(0,u.Z)(g);f=!1,s=l.Z,b=new o.Z}else b=e?[]:p;r:for(;++n<c;){var h=r[n],m=e?e(h):h;if(h=t||0!==h?h:0,f&&m==m){for(var v=b.length;v--;)if(b[v]===m)continue r;e&&b.push(m),p.push(h)}else s(b,m,t)||(b!==p&&b.push(m),p.push(h))}return p}},14658:function(r,e){e.Z=function(r,e){return r.has(e)}},64380:function(r,e){var t="object"==typeof global&&global&&global.Object===Object&&global;e.Z=t},47404:function(r,e,t){t.d(e,{Z:function(){return b}});var o,n=t(11146),i=t(48717).Z["__core-js_shared__"],a=(o=/[^.]+$/.exec(i&&i.keys&&i.keys.IE_PROTO||""))?"Symbol(src)_1."+o:"",l=t(84639),s=t(36423),c=/^\[object .+?Constructor\]$/,u=Object.prototype,d=Function.prototype.toString,f=u.hasOwnProperty,p=RegExp("^"+d.call(f).replace(/[\\^$.*+?()[\]{}|]/g,"\\$&").replace(/hasOwnProperty|(function).*?(?=\\\()| for .+?(?=\\\])/g,"$1.*?")+"$"),b=function(r,e){var t,o=null==r?void 0:r[e];return(t=o,(0,l.Z)(t)&&(!a||!(a in t))&&((0,n.Z)(t)?p:c).test((0,s.Z)(t)))?o:void 0}},48717:function(r,e,t){var o=t(64380),n="object"==typeof self&&self&&self.Object===Object&&self,i=o.Z||n||Function("return this")();e.Z=i},59410:function(r,e){e.Z=function(r){var e=-1,t=Array(r.size);return r.forEach(function(r){t[++e]=r}),t}},36423:function(r,e){var t=Function.prototype.toString;e.Z=function(r){if(null!=r){try{return t.call(r)}catch(r){}try{return r+""}catch(r){}}return""}},1853:function(r,e){e.Z=function(r){for(var e=-1,t=null==r?0:r.length,o=0,n=[];++e<t;){var i=r[e];i&&(n[o++]=i)}return n}},77725:function(r,e){e.Z=function(r,e){return r===e||r!=r&&e!=e}},11146:function(r,e,t){var o=t(17996),n=t(84639);e.Z=function(r){if(!(0,n.Z)(r))return!1;var e=(0,o.Z)(r);return"[object Function]"==e||"[object GeneratorFunction]"==e||"[object AsyncFunction]"==e||"[object Proxy]"==e}},74630:function(r,e){e.Z=function(r){return null==r}},84639:function(r,e){e.Z=function(r){var e=typeof r;return null!=r&&("object"==e||"function"==e)}},71480:function(r,e){e.Z=function(){}},52807:function(r,e,t){var o=t(19108);e.Z=function(r){return r&&r.length?(0,o.Z)(r):[]}},56575:function(r,e,t){t.d(e,{kP:function(){return i},x0:function(){return a}});let o=r=>crypto.getRandomValues(new Uint8Array(r)),n=(r,e,t)=>{let o=(2<<Math.log(r.length-1)/Math.LN2)-1,n=-~(1.6*o*e/r.length);return (i=e)=>{let a="";for(;;){let e=t(n),l=n;for(;l--;)if((a+=r[e[l]&o]||"").length===i)return a}}},i=(r,e=21)=>n(r,e,o),a=(r=21)=>crypto.getRandomValues(new Uint8Array(r)).reduce((r,e)=>((e&=63)<36?r+=e.toString(36):e<62?r+=(e-26).toString(36).toUpperCase():e>62?r+="-":r+="_",r),"")},48817:function(r,e,t){t.d(e,{m:function(){return E}});var o=/^\[(.+)\]$/;function n(r,e){var t=r;return e.split("-").forEach(function(r){t.nextPart.has(r)||t.nextPart.set(r,{nextPart:new Map,validators:[]}),t=t.nextPart.get(r)}),t}var i=/\s+/;function a(){for(var r,e,t=0,o="";t<arguments.length;)(r=arguments[t++])&&(e=function r(e){if("string"==typeof e)return e;for(var t,o="",n=0;n<e.length;n++)e[n]&&(t=r(e[n]))&&(o&&(o+=" "),o+=t);return o}(r))&&(o&&(o+=" "),o+=e);return o}function l(r){var e=function(e){return e[r]||[]};return e.isThemeGetter=!0,e}var s=/^\[(?:([a-z-]+):)?(.+)\]$/i,c=/^\d+\/\d+$/,u=new Set(["px","full","screen"]),d=/^(\d+(\.\d+)?)?(xs|sm|md|lg|xl)$/,f=/\d+(%|px|r?em|[sdl]?v([hwib]|min|max)|pt|pc|in|cm|mm|cap|ch|ex|r?lh|cq(w|h|i|b|min|max))|\b(calc|min|max|clamp)\(.+\)|^0$/,p=/^-?((\d+)?\.?(\d+)[a-z]+|0)_-?((\d+)?\.?(\d+)[a-z]+|0)/;function b(r){return x(r)||u.has(r)||c.test(r)||g(r)}function g(r){return O(r,"length",C)}function h(r){return O(r,"size",P)}function m(r){return O(r,"position",P)}function v(r){return O(r,"url",S)}function y(r){return O(r,"number",x)}function x(r){return!Number.isNaN(Number(r))}function w(r){return r.endsWith("%")&&x(r.slice(0,-1))}function _(r){return G(r)||O(r,"number",G)}function k(r){return s.test(r)}function Z(){return!0}function z(r){return d.test(r)}function j(r){return O(r,"",I)}function O(r,e,t){var o=s.exec(r);return!!o&&(o[1]?o[1]===e:t(o[2]))}function C(r){return f.test(r)}function P(){return!1}function S(r){return r.startsWith("url(")}function G(r){return Number.isInteger(Number(r))}function I(r){return p.test(r)}var E=function(){for(var r,e,t,l=arguments.length,s=Array(l),c=0;c<l;c++)s[c]=arguments[c];var u=function(i){var a=s[0];return e=(r=function(r){var e,t,i,a,l,s,c,u,d,f,p;return{cache:function(r){if(r<1)return{get:function(){},set:function(){}};var e=0,t=new Map,o=new Map;function n(n,i){t.set(n,i),++e>r&&(e=0,o=t,t=new Map)}return{get:function(r){var e=t.get(r);return void 0!==e?e:void 0!==(e=o.get(r))?(n(r,e),e):void 0},set:function(r,e){t.has(r)?t.set(r,e):n(r,e)}}}(r.cacheSize),splitModifiers:(t=1===(e=r.separator||":").length,i=e[0],a=e.length,function(r){for(var o,n=[],l=0,s=0,c=0;c<r.length;c++){var u=r[c];if(0===l){if(u===i&&(t||r.slice(c,c+a)===e)){n.push(r.slice(s,c)),s=c+a;continue}if("/"===u){o=c;continue}}"["===u?l++:"]"===u&&l--}var d=0===n.length?r:r.substring(s),f=d.startsWith("!"),p=f?d.substring(1):d;return{modifiers:n,hasImportantModifier:f,baseClassName:p,maybePostfixModifierPosition:o&&o>s?o-s:void 0}}),...(u=r.theme,d=r.prefix,f={nextPart:new Map,validators:[]},(p=Object.entries(r.classGroups),d?p.map(function(r){return[r[0],r[1].map(function(r){return"string"==typeof r?d+r:"object"==typeof r?Object.fromEntries(Object.entries(r).map(function(r){return[d+r[0],r[1]]})):r})]}):p).forEach(function(r){var e=r[0];(function r(e,t,o,i){e.forEach(function(e){if("string"==typeof e){(""===e?t:n(t,e)).classGroupId=o;return}if("function"==typeof e){if(e.isThemeGetter){r(e(i),t,o,i);return}t.validators.push({validator:e,classGroupId:o});return}Object.entries(e).forEach(function(e){var a=e[0];r(e[1],n(t,a),o,i)})})})(r[1],f,e,u)}),l=r.conflictingClassGroups,c=void 0===(s=r.conflictingClassGroupModifiers)?{}:s,{getClassGroupId:function(r){var e=r.split("-");return""===e[0]&&1!==e.length&&e.shift(),function r(e,t){if(0===e.length)return t.classGroupId;var o=e[0],n=t.nextPart.get(o),i=n?r(e.slice(1),n):void 0;if(i)return i;if(0!==t.validators.length){var a=e.join("-");return t.validators.find(function(r){return(0,r.validator)(a)})?.classGroupId}}(e,f)||function(r){if(o.test(r)){var e=o.exec(r)[1],t=e?.substring(0,e.indexOf(":"));if(t)return"arbitrary.."+t}}(r)},getConflictingClassGroupIds:function(r,e){var t=l[r]||[];return e&&c[r]?[].concat(t,c[r]):t}})}}(s.slice(1).reduce(function(r,e){return e(r)},a()))).cache.get,t=r.cache.set,u=d,d(i)};function d(o){var n,a,l,s,c,u=e(o);if(u)return u;var d=(a=(n=r).splitModifiers,l=n.getClassGroupId,s=n.getConflictingClassGroupIds,c=new Set,o.trim().split(i).map(function(r){var e=a(r),t=e.modifiers,o=e.hasImportantModifier,n=e.baseClassName,i=e.maybePostfixModifierPosition,s=l(i?n.substring(0,i):n),c=!!i;if(!s){if(!i||!(s=l(n)))return{isTailwindClass:!1,originalClassName:r};c=!1}var u=(function(r){if(r.length<=1)return r;var e=[],t=[];return r.forEach(function(r){"["===r[0]?(e.push.apply(e,t.sort().concat([r])),t=[]):t.push(r)}),e.push.apply(e,t.sort()),e})(t).join(":");return{isTailwindClass:!0,modifierId:o?u+"!":u,classGroupId:s,originalClassName:r,hasPostfixModifier:c}}).reverse().filter(function(r){if(!r.isTailwindClass)return!0;var e=r.modifierId,t=r.classGroupId,o=r.hasPostfixModifier,n=e+t;return!c.has(n)&&(c.add(n),s(t,o).forEach(function(r){return c.add(e+r)}),!0)}).reverse().map(function(r){return r.originalClassName}).join(" "));return t(o,d),d}return function(){return u(a.apply(null,arguments))}}(function(){var r=l("colors"),e=l("spacing"),t=l("blur"),o=l("brightness"),n=l("borderColor"),i=l("borderRadius"),a=l("borderSpacing"),s=l("borderWidth"),c=l("contrast"),u=l("grayscale"),d=l("hueRotate"),f=l("invert"),p=l("gap"),O=l("gradientColorStops"),C=l("gradientColorStopPositions"),P=l("inset"),S=l("margin"),G=l("opacity"),I=l("padding"),E=l("saturate"),M=l("scale"),N=l("sepia"),R=l("skew"),$=l("space"),T=l("translate"),A=function(){return["auto","contain","none"]},F=function(){return["auto","hidden","clip","visible","scroll"]},W=function(){return["auto",k,e]},U=function(){return[k,e]},L=function(){return["",b]},q=function(){return["auto",x,k]},D=function(){return["bottom","center","left","left-bottom","left-top","right","right-bottom","right-top","top"]},V=function(){return["solid","dashed","dotted","double","none"]},B=function(){return["normal","multiply","screen","overlay","darken","lighten","color-dodge","color-burn","hard-light","soft-light","difference","exclusion","hue","saturation","color","luminosity","plus-lighter"]},Y=function(){return["start","end","center","between","around","evenly","stretch"]},H=function(){return["","0",k]},J=function(){return["auto","avoid","all","avoid-page","page","left","right","column"]},K=function(){return[x,y]},Q=function(){return[x,k]};return{cacheSize:500,theme:{colors:[Z],spacing:[b],blur:["none","",z,k],brightness:K(),borderColor:[r],borderRadius:["none","","full",z,k],borderSpacing:U(),borderWidth:L(),contrast:K(),grayscale:H(),hueRotate:Q(),invert:H(),gap:U(),gradientColorStops:[r],gradientColorStopPositions:[w,g],inset:W(),margin:W(),opacity:K(),padding:U(),saturate:K(),scale:K(),sepia:H(),skew:Q(),space:U(),translate:U()},classGroups:{aspect:[{aspect:["auto","square","video",k]}],container:["container"],columns:[{columns:[z]}],"break-after":[{"break-after":J()}],"break-before":[{"break-before":J()}],"break-inside":[{"break-inside":["auto","avoid","avoid-page","avoid-column"]}],"box-decoration":[{"box-decoration":["slice","clone"]}],box:[{box:["border","content"]}],display:["block","inline-block","inline","flex","inline-flex","table","inline-table","table-caption","table-cell","table-column","table-column-group","table-footer-group","table-header-group","table-row-group","table-row","flow-root","grid","inline-grid","contents","list-item","hidden"],float:[{float:["right","left","none"]}],clear:[{clear:["left","right","both","none"]}],isolation:["isolate","isolation-auto"],"object-fit":[{object:["contain","cover","fill","none","scale-down"]}],"object-position":[{object:[].concat(D(),[k])}],overflow:[{overflow:F()}],"overflow-x":[{"overflow-x":F()}],"overflow-y":[{"overflow-y":F()}],overscroll:[{overscroll:A()}],"overscroll-x":[{"overscroll-x":A()}],"overscroll-y":[{"overscroll-y":A()}],position:["static","fixed","absolute","relative","sticky"],inset:[{inset:[P]}],"inset-x":[{"inset-x":[P]}],"inset-y":[{"inset-y":[P]}],start:[{start:[P]}],end:[{end:[P]}],top:[{top:[P]}],right:[{right:[P]}],bottom:[{bottom:[P]}],left:[{left:[P]}],visibility:["visible","invisible","collapse"],z:[{z:["auto",_]}],basis:[{basis:W()}],"flex-direction":[{flex:["row","row-reverse","col","col-reverse"]}],"flex-wrap":[{flex:["wrap","wrap-reverse","nowrap"]}],flex:[{flex:["1","auto","initial","none",k]}],grow:[{grow:H()}],shrink:[{shrink:H()}],order:[{order:["first","last","none",_]}],"grid-cols":[{"grid-cols":[Z]}],"col-start-end":[{col:["auto",{span:["full",_]},k]}],"col-start":[{"col-start":q()}],"col-end":[{"col-end":q()}],"grid-rows":[{"grid-rows":[Z]}],"row-start-end":[{row:["auto",{span:[_]},k]}],"row-start":[{"row-start":q()}],"row-end":[{"row-end":q()}],"grid-flow":[{"grid-flow":["row","col","dense","row-dense","col-dense"]}],"auto-cols":[{"auto-cols":["auto","min","max","fr",k]}],"auto-rows":[{"auto-rows":["auto","min","max","fr",k]}],gap:[{gap:[p]}],"gap-x":[{"gap-x":[p]}],"gap-y":[{"gap-y":[p]}],"justify-content":[{justify:["normal"].concat(Y())}],"justify-items":[{"justify-items":["start","end","center","stretch"]}],"justify-self":[{"justify-self":["auto","start","end","center","stretch"]}],"align-content":[{content:["normal"].concat(Y(),["baseline"])}],"align-items":[{items:["start","end","center","baseline","stretch"]}],"align-self":[{self:["auto","start","end","center","stretch","baseline"]}],"place-content":[{"place-content":[].concat(Y(),["baseline"])}],"place-items":[{"place-items":["start","end","center","baseline","stretch"]}],"place-self":[{"place-self":["auto","start","end","center","stretch"]}],p:[{p:[I]}],px:[{px:[I]}],py:[{py:[I]}],ps:[{ps:[I]}],pe:[{pe:[I]}],pt:[{pt:[I]}],pr:[{pr:[I]}],pb:[{pb:[I]}],pl:[{pl:[I]}],m:[{m:[S]}],mx:[{mx:[S]}],my:[{my:[S]}],ms:[{ms:[S]}],me:[{me:[S]}],mt:[{mt:[S]}],mr:[{mr:[S]}],mb:[{mb:[S]}],ml:[{ml:[S]}],"space-x":[{"space-x":[$]}],"space-x-reverse":["space-x-reverse"],"space-y":[{"space-y":[$]}],"space-y-reverse":["space-y-reverse"],w:[{w:["auto","min","max","fit",k,e]}],"min-w":[{"min-w":["min","max","fit",k,b]}],"max-w":[{"max-w":["0","none","full","min","max","fit","prose",{screen:[z]},z,k]}],h:[{h:[k,e,"auto","min","max","fit"]}],"min-h":[{"min-h":["min","max","fit",k,b]}],"max-h":[{"max-h":[k,e,"min","max","fit"]}],"font-size":[{text:["base",z,g]}],"font-smoothing":["antialiased","subpixel-antialiased"],"font-style":["italic","not-italic"],"font-weight":[{font:["thin","extralight","light","normal","medium","semibold","bold","extrabold","black",y]}],"font-family":[{font:[Z]}],"fvn-normal":["normal-nums"],"fvn-ordinal":["ordinal"],"fvn-slashed-zero":["slashed-zero"],"fvn-figure":["lining-nums","oldstyle-nums"],"fvn-spacing":["proportional-nums","tabular-nums"],"fvn-fraction":["diagonal-fractions","stacked-fractons"],tracking:[{tracking:["tighter","tight","normal","wide","wider","widest",k]}],"line-clamp":[{"line-clamp":["none",x,y]}],leading:[{leading:["none","tight","snug","normal","relaxed","loose",k,b]}],"list-image":[{"list-image":["none",k]}],"list-style-type":[{list:["none","disc","decimal",k]}],"list-style-position":[{list:["inside","outside"]}],"placeholder-color":[{placeholder:[r]}],"placeholder-opacity":[{"placeholder-opacity":[G]}],"text-alignment":[{text:["left","center","right","justify","start","end"]}],"text-color":[{text:[r]}],"text-opacity":[{"text-opacity":[G]}],"text-decoration":["underline","overline","line-through","no-underline"],"text-decoration-style":[{decoration:[].concat(V(),["wavy"])}],"text-decoration-thickness":[{decoration:["auto","from-font",b]}],"underline-offset":[{"underline-offset":["auto",k,b]}],"text-decoration-color":[{decoration:[r]}],"text-transform":["uppercase","lowercase","capitalize","normal-case"],"text-overflow":["truncate","text-ellipsis","text-clip"],indent:[{indent:U()}],"vertical-align":[{align:["baseline","top","middle","bottom","text-top","text-bottom","sub","super",k]}],whitespace:[{whitespace:["normal","nowrap","pre","pre-line","pre-wrap","break-spaces"]}],break:[{break:["normal","words","all","keep"]}],hyphens:[{hyphens:["none","manual","auto"]}],content:[{content:["none",k]}],"bg-attachment":[{bg:["fixed","local","scroll"]}],"bg-clip":[{"bg-clip":["border","padding","content","text"]}],"bg-opacity":[{"bg-opacity":[G]}],"bg-origin":[{"bg-origin":["border","padding","content"]}],"bg-position":[{bg:[].concat(D(),[m])}],"bg-repeat":[{bg:["no-repeat",{repeat:["","x","y","round","space"]}]}],"bg-size":[{bg:["auto","cover","contain",h]}],"bg-image":[{bg:["none",{"gradient-to":["t","tr","r","br","b","bl","l","tl"]},v]}],"bg-color":[{bg:[r]}],"gradient-from-pos":[{from:[C]}],"gradient-via-pos":[{via:[C]}],"gradient-to-pos":[{to:[C]}],"gradient-from":[{from:[O]}],"gradient-via":[{via:[O]}],"gradient-to":[{to:[O]}],rounded:[{rounded:[i]}],"rounded-s":[{"rounded-s":[i]}],"rounded-e":[{"rounded-e":[i]}],"rounded-t":[{"rounded-t":[i]}],"rounded-r":[{"rounded-r":[i]}],"rounded-b":[{"rounded-b":[i]}],"rounded-l":[{"rounded-l":[i]}],"rounded-ss":[{"rounded-ss":[i]}],"rounded-se":[{"rounded-se":[i]}],"rounded-ee":[{"rounded-ee":[i]}],"rounded-es":[{"rounded-es":[i]}],"rounded-tl":[{"rounded-tl":[i]}],"rounded-tr":[{"rounded-tr":[i]}],"rounded-br":[{"rounded-br":[i]}],"rounded-bl":[{"rounded-bl":[i]}],"border-w":[{border:[s]}],"border-w-x":[{"border-x":[s]}],"border-w-y":[{"border-y":[s]}],"border-w-s":[{"border-s":[s]}],"border-w-e":[{"border-e":[s]}],"border-w-t":[{"border-t":[s]}],"border-w-r":[{"border-r":[s]}],"border-w-b":[{"border-b":[s]}],"border-w-l":[{"border-l":[s]}],"border-opacity":[{"border-opacity":[G]}],"border-style":[{border:[].concat(V(),["hidden"])}],"divide-x":[{"divide-x":[s]}],"divide-x-reverse":["divide-x-reverse"],"divide-y":[{"divide-y":[s]}],"divide-y-reverse":["divide-y-reverse"],"divide-opacity":[{"divide-opacity":[G]}],"divide-style":[{divide:V()}],"border-color":[{border:[n]}],"border-color-x":[{"border-x":[n]}],"border-color-y":[{"border-y":[n]}],"border-color-t":[{"border-t":[n]}],"border-color-r":[{"border-r":[n]}],"border-color-b":[{"border-b":[n]}],"border-color-l":[{"border-l":[n]}],"divide-color":[{divide:[n]}],"outline-style":[{outline:[""].concat(V())}],"outline-offset":[{"outline-offset":[k,b]}],"outline-w":[{outline:[b]}],"outline-color":[{outline:[r]}],"ring-w":[{ring:L()}],"ring-w-inset":["ring-inset"],"ring-color":[{ring:[r]}],"ring-opacity":[{"ring-opacity":[G]}],"ring-offset-w":[{"ring-offset":[b]}],"ring-offset-color":[{"ring-offset":[r]}],shadow:[{shadow:["","inner","none",z,j]}],"shadow-color":[{shadow:[Z]}],opacity:[{opacity:[G]}],"mix-blend":[{"mix-blend":B()}],"bg-blend":[{"bg-blend":B()}],filter:[{filter:["","none"]}],blur:[{blur:[t]}],brightness:[{brightness:[o]}],contrast:[{contrast:[c]}],"drop-shadow":[{"drop-shadow":["","none",z,k]}],grayscale:[{grayscale:[u]}],"hue-rotate":[{"hue-rotate":[d]}],invert:[{invert:[f]}],saturate:[{saturate:[E]}],sepia:[{sepia:[N]}],"backdrop-filter":[{"backdrop-filter":["","none"]}],"backdrop-blur":[{"backdrop-blur":[t]}],"backdrop-brightness":[{"backdrop-brightness":[o]}],"backdrop-contrast":[{"backdrop-contrast":[c]}],"backdrop-grayscale":[{"backdrop-grayscale":[u]}],"backdrop-hue-rotate":[{"backdrop-hue-rotate":[d]}],"backdrop-invert":[{"backdrop-invert":[f]}],"backdrop-opacity":[{"backdrop-opacity":[G]}],"backdrop-saturate":[{"backdrop-saturate":[E]}],"backdrop-sepia":[{"backdrop-sepia":[N]}],"border-collapse":[{border:["collapse","separate"]}],"border-spacing":[{"border-spacing":[a]}],"border-spacing-x":[{"border-spacing-x":[a]}],"border-spacing-y":[{"border-spacing-y":[a]}],"table-layout":[{table:["auto","fixed"]}],caption:[{caption:["top","bottom"]}],transition:[{transition:["none","all","","colors","opacity","shadow","transform",k]}],duration:[{duration:Q()}],ease:[{ease:["linear","in","out","in-out",k]}],delay:[{delay:Q()}],animate:[{animate:["none","spin","ping","pulse","bounce",k]}],transform:[{transform:["","gpu","none"]}],scale:[{scale:[M]}],"scale-x":[{"scale-x":[M]}],"scale-y":[{"scale-y":[M]}],rotate:[{rotate:[_,k]}],"translate-x":[{"translate-x":[T]}],"translate-y":[{"translate-y":[T]}],"skew-x":[{"skew-x":[R]}],"skew-y":[{"skew-y":[R]}],"transform-origin":[{origin:["center","top","top-right","right","bottom-right","bottom","bottom-left","left","top-left",k]}],accent:[{accent:["auto",r]}],appearance:["appearance-none"],cursor:[{cursor:["auto","default","pointer","wait","text","move","help","not-allowed","none","context-menu","progress","cell","crosshair","vertical-text","alias","copy","no-drop","grab","grabbing","all-scroll","col-resize","row-resize","n-resize","e-resize","s-resize","w-resize","ne-resize","nw-resize","se-resize","sw-resize","ew-resize","ns-resize","nesw-resize","nwse-resize","zoom-in","zoom-out",k]}],"caret-color":[{caret:[r]}],"pointer-events":[{"pointer-events":["none","auto"]}],resize:[{resize:["none","y","x",""]}],"scroll-behavior":[{scroll:["auto","smooth"]}],"scroll-m":[{"scroll-m":U()}],"scroll-mx":[{"scroll-mx":U()}],"scroll-my":[{"scroll-my":U()}],"scroll-ms":[{"scroll-ms":U()}],"scroll-me":[{"scroll-me":U()}],"scroll-mt":[{"scroll-mt":U()}],"scroll-mr":[{"scroll-mr":U()}],"scroll-mb":[{"scroll-mb":U()}],"scroll-ml":[{"scroll-ml":U()}],"scroll-p":[{"scroll-p":U()}],"scroll-px":[{"scroll-px":U()}],"scroll-py":[{"scroll-py":U()}],"scroll-ps":[{"scroll-ps":U()}],"scroll-pe":[{"scroll-pe":U()}],"scroll-pt":[{"scroll-pt":U()}],"scroll-pr":[{"scroll-pr":U()}],"scroll-pb":[{"scroll-pb":U()}],"scroll-pl":[{"scroll-pl":U()}],"snap-align":[{snap:["start","end","center","align-none"]}],"snap-stop":[{snap:["normal","always"]}],"snap-type":[{snap:["none","x","y","both"]}],"snap-strictness":[{snap:["mandatory","proximity"]}],touch:[{touch:["auto","none","pinch-zoom","manipulation",{pan:["x","left","right","y","up","down"]}]}],select:[{select:["none","text","all","auto"]}],"will-change":[{"will-change":["auto","scroll","contents","transform",k]}],fill:[{fill:[r,"none"]}],"stroke-w":[{stroke:[b,y]}],stroke:[{stroke:[r,"none"]}],sr:["sr-only","not-sr-only"]},conflictingClassGroups:{overflow:["overflow-x","overflow-y"],overscroll:["overscroll-x","overscroll-y"],inset:["inset-x","inset-y","start","end","top","right","bottom","left"],"inset-x":["right","left"],"inset-y":["top","bottom"],flex:["basis","grow","shrink"],gap:["gap-x","gap-y"],p:["px","py","ps","pe","pt","pr","pb","pl"],px:["pr","pl"],py:["pt","pb"],m:["mx","my","ms","me","mt","mr","mb","ml"],mx:["mr","ml"],my:["mt","mb"],"font-size":["leading"],"fvn-normal":["fvn-ordinal","fvn-slashed-zero","fvn-figure","fvn-spacing","fvn-fraction"],"fvn-ordinal":["fvn-normal"],"fvn-slashed-zero":["fvn-normal"],"fvn-figure":["fvn-normal"],"fvn-spacing":["fvn-normal"],"fvn-fraction":["fvn-normal"],rounded:["rounded-s","rounded-e","rounded-t","rounded-r","rounded-b","rounded-l","rounded-ss","rounded-se","rounded-ee","rounded-es","rounded-tl","rounded-tr","rounded-br","rounded-bl"],"rounded-s":["rounded-ss","rounded-es"],"rounded-e":["rounded-se","rounded-ee"],"rounded-t":["rounded-tl","rounded-tr"],"rounded-r":["rounded-tr","rounded-br"],"rounded-b":["rounded-br","rounded-bl"],"rounded-l":["rounded-tl","rounded-bl"],"border-spacing":["border-spacing-x","border-spacing-y"],"border-w":["border-w-s","border-w-e","border-w-t","border-w-r","border-w-b","border-w-l"],"border-w-x":["border-w-r","border-w-l"],"border-w-y":["border-w-t","border-w-b"],"border-color":["border-color-t","border-color-r","border-color-b","border-color-l"],"border-color-x":["border-color-r","border-color-l"],"border-color-y":["border-color-t","border-color-b"],"scroll-m":["scroll-mx","scroll-my","scroll-ms","scroll-me","scroll-mt","scroll-mr","scroll-mb","scroll-ml"],"scroll-mx":["scroll-mr","scroll-ml"],"scroll-my":["scroll-mt","scroll-mb"],"scroll-p":["scroll-px","scroll-py","scroll-ps","scroll-pe","scroll-pt","scroll-pr","scroll-pb","scroll-pl"],"scroll-px":["scroll-pr","scroll-pl"],"scroll-py":["scroll-pt","scroll-pb"]},conflictingClassGroupModifiers:{"font-size":["leading"]}}})}}]);