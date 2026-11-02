"""Real Codex inference; validated schedule only, own-arm observations, no tools."""
from campaign_support import *
from ai_supervisor import PROMPT,SCHEMA,validate
import subprocess,shutil

class InferenceInterrupted(Exception):pass

def command(exe,schema,output,work):
    args=[exe,'--no-daemon','-a','never','exec','--ignore-user-config','--ephemeral',
          '--skip-git-repo-check','--sandbox','read-only','--json','--color','never',
          '-C',str(work),'--output-schema',str(schema),'-o',str(output)]
    for name in ('shell_tool','apps','multi_agent','hooks','memories','plugins','browser_use','computer_use'):
        args+=['-c',f'features.{name}=false']
    args+=['-c','web_search="disabled"','-c','mcp_servers={}',
           '-c','developer_instructions='+json.dumps(PROMPT+' Do not call any tools. Do not read files. Use only the supplied JSON observation. No messages or notifications. Return only the required JSON.'),'-']
    return args

class CodexSupervisor:
    def __init__(self,path):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
        self.calls=len(list(self.path.glob('*.request.json')));self.wait_s=0.;self.usage=[]
        self.exe=shutil.which('codex')
        if not self.exe:raise RuntimeError('Codex CLI not found; no substitute AI policy')
    def choose(self,observation,stopped=lambda:False,heartbeat=lambda:None):
        ix=self.calls;self.calls+=1;prefix=self.path/f'{ix:06d}';started=time.monotonic()
        request=dict(schema_version=1,instructions=PROMPT,observation=observation)
        h=hashlib.sha256(json.dumps(request,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        save(prefix.with_suffix('.request.json'),dict(request_sha256=h,**request))
        schema=prefix.with_suffix('.schema.json');save(schema,SCHEMA)
        out=prefix.with_suffix('.response.json');cwd=self.path/'empty_context';cwd.mkdir(exist_ok=True)
        args=command(self.exe,schema.resolve(),out.resolve(),cwd.resolve())
        save(prefix.with_suffix('.command.json'),dict(args=args,created_utc=now(),request_sha256=h,role='schedule inference only'))
        events=prefix.with_suffix('.events.jsonl');errors=prefix.with_suffix('.stderr.log')
        try:
            with events.open('w',encoding='utf-8') as stdout,errors.open('w',encoding='utf-8') as stderr:
                p=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=stdout,stderr=stderr,
                    text=True,encoding='utf-8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                p.stdin.write(json.dumps(request,ensure_ascii=False));p.stdin.close()
                last=0.
                while p.poll() is None:
                    if stopped() or time.monotonic()-started>240:
                        p.terminate();p.wait(timeout=15)
                        if stopped():raise InferenceInterrupted('Pair stopped during model request')
                        raise TimeoutError('Supervisor inference exceeded 240 seconds')
                    if time.monotonic()-last>=10:heartbeat();last=time.monotonic()
                    time.sleep(.5)
            if p.returncode!=0:raise RuntimeError('Codex inference failed with exit '+str(p.returncode)+'; see '+str(errors))
            if not out.exists():raise RuntimeError('No structured model response')
            decision=validate(read(out));usage=[];tool_events=[]
            for line in events.read_text('utf-8').splitlines():
                try:e=json.loads(line)
                except json.JSONDecodeError:continue
                if e.get('usage'):usage.append(e['usage'])
                if e.get('item',{}).get('type') in ('command_execution','mcp_tool_call','web_search','file_change'):
                    tool_events.append(e)
            if tool_events:raise RuntimeError('Supervisor used forbidden tools; decision rejected')
            elapsed=time.monotonic()-started;self.usage.extend(usage)
            save(prefix.with_suffix('.accepted.json'),dict(provider='codex_cli_authenticated_chatgpt',request_sha256=h,
                decision=decision,elapsed_s=elapsed,usage=usage,tool_events=tool_events,
                model='CLI default; model identifier is recorded in stderr when exposed',response_sha256=sha(out)))
            return decision
        except Exception as e:
            save(prefix.with_suffix('.error.json'),dict(error=str(e),request_sha256=h,utc=now()))
            raise
        finally:self.wait_s+=time.monotonic()-started

if __name__=='__main__':
    from hybrid_core import FAMILIES
    observation=dict(test_only=True,phase='strict',families=FAMILIES,calls=8,current=read(D/'preflight/summary.json')['rows'][0]['result'],
                     rules='A transport test. Choose a valid schedule for this observation. No tools.')
    s=CodexSupervisor(D/'preflight/ai_live');print(json.dumps(s.choose(observation)),flush=True)
