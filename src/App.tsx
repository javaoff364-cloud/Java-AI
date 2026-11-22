import { useEffect, useRef, useState } from "react";
import { ArrowUp, Menu, MessageSquare, Plus, Sparkles, Sun, Moon, Trash2, X } from "lucide-react";

type Message = { role: "user" | "assistant"; content: string };
type Chat = { id: string; title: string; messages: Message[] };
const suggestions = [
  ["✦","Tushuntirib ber","Murakkab mavzuni oddiy o'zbek tilida tushuntir."],
  ["⌘","Kod yozish","Python tilida oddiy kalkulyator dasturini yozib ber."],
  ["↗","Matn yozish","Menga chiroyli va tushunarli loyiha tavsifi yozib ber."],
  ["◎","Til o'rganish","Ingliz tilida 5 ta foydali ibora va tarjimasini ber."]
];

export default function Home() {
  const [chats,setChats] = useState<Chat[]>([]);
  const [activeId,setActiveId] = useState("");
  const [input,setInput] = useState("");
  const [loading,setLoading] = useState(false);
  const [dark,setDark] = useState(true);
  const [sidebar,setSidebar] = useState(true);
  const [error,setError] = useState("");
  const bottom = useRef<HTMLDivElement>(null);
  const active = chats.find(c => c.id === activeId);

  useEffect(() => {
    try {
      const saved = localStorage.getItem("javus-ai-chats");
      if (saved) { const parsed = JSON.parse(saved) as Chat[]; setChats(parsed); if (parsed[0]) setActiveId(parsed[0].id); }
      setDark(localStorage.getItem("javus-ai-theme") !== "light");
    } catch {}
  },[]);
  useEffect(() => { localStorage.setItem("javus-ai-chats",JSON.stringify(chats)); },[chats]);
  useEffect(() => { localStorage.setItem("javus-ai-theme",dark?"dark":"light"); },[dark]);
  useEffect(() => { bottom.current?.scrollIntoView({behavior:"smooth"}); },[active?.messages,loading]);

  function newChat() { const id=crypto.randomUUID(); setChats(old=>[{id,title:"Yangi suhbat",messages:[]},...old]); setActiveId(id); setError(""); }
  function deleteChat(id:string) { const next=chats.filter(c=>c.id!==id); setChats(next); if(activeId===id)setActiveId(next[0]?.id||""); }

  async function send(text=input) {
    const prompt=text.trim(); if(!prompt||loading)return;
    let chat=active;
    if(!chat) { const id=crypto.randomUUID(); chat={id,title:prompt.slice(0,32),messages:[]}; setChats(old=>[{...chat!},...old]); setActiveId(id); }
    const id=chat.id;
    const history=[...chat.messages,{role:"user" as const,content:prompt}];
    setChats(old=>old.map(c=>c.id===id?{...c,title:c.messages.length?c.title:prompt.slice(0,32),messages:history}:c));
    setInput(""); setLoading(true); setError("");
    try {
      const res=await fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({messages:history})});
      const data=await res.json(); if(!res.ok)throw new Error(data.error||data.detail||"Xatolik yuz berdi.");
      setChats(old=>old.map(c=>c.id===id?{...c,messages:[...c.messages,{role:"assistant",content:data.reply}]}:c));
    } catch(e) { setError(e instanceof Error?e.message:"Javus AI bilan bog'lanib bo'lmadi."); }
    finally { setLoading(false); }
  }

  return <main className={dark?"app dark":"app light"}>
    <aside className={sidebar?"sidebar":"sidebar closed"}>
      <div className="brand"><img className="brand-logo" src="/javus-mark.svg" alt="Javus AI logo"/><span>Javus<span className="ai">AI</span></span><button className="sidebar-close" onClick={()=>setSidebar(false)} aria-label="Yon panelni yopish"><X size={18}/></button></div>
      <button className="new-chat" onClick={newChat}><Plus size={17}/> Yangi chat</button>
      <div className="label">SUHBATLAR</div>
      <div className="chat-list">{chats.map(c=><div className={"chat-item "+(c.id===activeId?"selected":"")} key={c.id}>
        <button className="chat-name" onClick={()=>setActiveId(c.id)}><MessageSquare size={15}/><span>{c.title}</span></button>
        <button className="delete" onClick={()=>deleteChat(c.id)} aria-label="Chatni o'chirish"><Trash2 size={14}/></button>
      </div>)}</div>
      <div className="profile"><div className="avatar"><img src="/javus-mark.svg" alt="Javus AI"/></div><div><b>Javus AI</b><small>Experimental neural model</small></div><button onClick={()=>setDark(!dark)} aria-label="Mavzuni almashtirish">{dark?<Sun size={17}/>:<Moon size={17}/>}</button></div>
    </aside>
    {sidebar&&<button className="sidebar-backdrop" onClick={()=>setSidebar(false)} aria-label="Yon panelni yopish"/>}
    <section className="main">
      <header className="topbar"><button className="icon" onClick={()=>setSidebar(!sidebar)} aria-label="Sidebar"><Menu size={19}/></button><div className="model"><i/> Javus <span>· Core 0.1</span></div><span className="beta">EXPERIMENTAL</span></header>
      <div className="conversation">
        {!active?.messages.length?<div className="welcome"><div className="hero-icon"><img src="/javus-mark.svg" alt="Javus AI"/></div><div className="eyebrow">JAVUS AI · NEURAL NETWORK</div><h1>Yangi bilimlarni<br/><em>birga o'rganamiz.</em></h1><p>Javus o'zimiz yaratgan tajribaviy neyron tarmoq. Hozircha kichik datasetda o'qitilmoqda.</p><div className="suggestions">{suggestions.map(([icon,title,desc])=><button key={title} onClick={()=>send(desc)}><span>{icon}</span><div><b>{title}</b><small>{desc}</small></div><ArrowUp size={15}/></button>)}</div></div>:<div className="messages">{active.messages.map((m,i)=><div className={"message "+m.role} key={i}>{m.role==="assistant"&&<div className="message-avatar"><Sparkles size={15}/></div>}<div className="bubble"><small>{m.role==="user"?"Siz":"Javus"}</small><div>{m.content}</div></div></div>)}{loading&&<div className="message assistant"><div className="message-avatar"><Sparkles size={15}/></div><div className="bubble"><small>Javus</small><div className="dots">● ● ●</div></div></div>}<div ref={bottom}/></div>}
      </div>
      {error&&<div className="error">{error}</div>}
      <div className="composer-wrap"><form className="composer" onSubmit={e=>{e.preventDefault();send();}}><textarea value={input} onChange={e=>setInput(e.target.value)} onKeyDown={e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send();}}} placeholder="Javus AI'ga xabar yuboring..." rows={1}/><div className="composer-bottom"><span>Javus tajribaviy model. Javoblar xato bo'lishi mumkin.</span><button disabled={!input.trim()||loading} aria-label="Yuborish"><ArrowUp size={19}/></button></div></form><div className="foot">O'z neyron tarmog'imiz · Javus AI</div></div>
    </section>
  </main>;
}
