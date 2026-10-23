// Meaning search: a WordPiece tokenizer and mean-pooled static embeddings (model2vec "potion") in plain JS.
// Mirrors model2vec's encode(): BERT normalisation + WordPiece, no special tokens, unknown tokens dropped,
// mean of token vectors, L2-normalised. Model and item vectors are int8 with one float32 scale per row.
const Meaning = (function(){
  let model=null, items=null, loading=null;
  const PUNCT=/[\p{P}$+<=>^`|~]/u;
  function basicTokens(text){
    let s=String(text).normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase();
    s=s.replace(/[\u0000\ufffd]/g,'').replace(/[\u4e00-\u9fff\u3400-\u4dbf]/g,c=>' '+c+' ');
    const out=[];
    for(const w of s.split(/\s+/)){ if(!w) continue; let cur='';
      for(const ch of w){ if(PUNCT.test(ch)){ if(cur){out.push(cur);cur='';} out.push(ch);} else cur+=ch; }
      if(cur) out.push(cur); }
    return out;
  }
  function wordpiece(word, vocab, unk){
    if(word.length>100) return [unk];
    const ids=[]; let start=0;
    while(start<word.length){ let end=word.length, found=-1;
      while(start<end){ const sub=(start>0?'##':'')+word.slice(start,end); if(vocab[sub]!==undefined){found=vocab[sub];break;} end--; }
      if(found<0) return [unk];
      ids.push(found); start=end; }
    return ids;
  }
  function tokenIds(text){ const ids=[]; for(const w of basicTokens(text)) for(const id of wordpiece(w, model.vocab, model.unk_id)) if(id!==model.unk_id) ids.push(id); return ids; }
  function encode(text){
    const d=model.dim, v=new Float32Array(d); const ids=tokenIds(text); if(!ids.length) return v;
    for(const id of ids){ const sc=model.scales[id], off=id*d; for(let k=0;k<d;k++) v[k]+=model.table[off+k]*sc; }
    let n=0; for(let k=0;k<d;k++){ v[k]/=ids.length; n+=v[k]*v[k]; } n=Math.sqrt(n)||1; for(let k=0;k<d;k++) v[k]/=n; return v;
  }
  async function load(onProgress){
    if(model&&items) return true; if(loading) return loading;
    loading=(async()=>{ try{
      onProgress&&onProgress('Loading the meaning-search model (about 18 MB, once)');
      const b64=r=>r.text().then(s=>{ const bin=atob(s.trim()); const u=new Uint8Array(bin.length); for(let i=0;i<bin.length;i++) u[i]=bin.charCodeAt(i); return u.buffer; });
      const [mj,mb,ms,ej,eb,es]=await Promise.all([fetch('model.json').then(r=>r.json()),fetch('model.bin.txt').then(b64),fetch('model_scales.bin.txt').then(b64),fetch('emb.json').then(r=>r.json()),fetch('emb.bin.txt').then(b64),fetch('emb_scales.bin.txt').then(b64)]);
      model={vocab:mj.vocab,dim:mj.dim,unk_id:mj.unk_id,table:new Int8Array(mb),scales:new Float32Array(ms)};
      items={n:ej.n,dim:ej.dim,refs:ej.refs,vec:new Int8Array(eb),scales:new Float32Array(es)};
      return true; }catch(e){ loading=null; return false; } })();
    return loading;
  }
  function search(text, k){
    if(!model||!items) return [];
    const q=encode(text); const d=items.dim, n=items.n, vec=items.vec, sc=items.scales;
    const scores=new Float32Array(n);
    for(let i=0;i<n;i++){ let s=0; const off=i*d; for(let j=0;j<d;j++) s+=q[j]*vec[off+j]; scores[i]=s*sc[i]; }
    const idx=[]; for(let i=0;i<n;i++) idx.push(i);
    idx.sort((a,b)=>scores[b]-scores[a]);
    return idx.slice(0,k).map(i=>({ref:items.refs[i],score:scores[i]}));
  }
  return {load, encode, search, tokenIds, ready:()=>!!(model&&items), _setModel:(m)=>{model=m;}};
})();
if(typeof module!=='undefined') module.exports=Meaning;
