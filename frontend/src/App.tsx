import React,{useState,useEffect,FormEvent} from 'react';import axios from 'axios';import './App.css';
interface Task{id:number;prompt:string;status:string;result_a?:string;result_b?:string;}
const API_URL='http://localhost:8000';
function App(){
  const [tasks,setTasks]=useState<Task[]>([]);const [prompt,setPrompt]=useState('');
  const handleCreateTask=async (e:FormEvent)=>{
    e.preventDefault();if(!prompt.trim())return;
    const {data}=await axios.post<Task>(`${API_URL}/tasks/`,{prompt});
    setTasks(p=>[...p,data]);setPrompt('');
  };
  return(
    <div className="App"><h1>🧠 AI Command Center</h1>
      <form onSubmit={handleCreateTask}><input value={prompt} onChange={e=>setPrompt(e.target.value)} placeholder="Wpisz prompt..."/><button type="submit">Uruchom</button></form>
      <div className="task-list">
        {tasks.map(t=>(<div key={t.id} className="task-item"><h3>{t.prompt}</h3><p>Status: {t.status}</p>{t.result_a&&<div><strong>A:</strong> {t.result_a}</div>}{t.result_b&&<div><strong>B:</strong> {t.result_b}</div>}</div>))}
      </div>
    </div>
  );
}
export default App;
