
import subprocess, json

def pylint_score(path='.'):
    proc=subprocess.run(['pylint','--score','y',path,'-sn','-rn','--exit-zero'],capture_output=True,text=True)
    out=proc.stdout.split('
')[-2]
    score=float(out.split('/')[0].split()[-1]) if '/' in out else 0.0
    return score

def black_check(path='.'):
    proc=subprocess.run(['black','--check','--diff',path],capture_output=True,text=True)
    return proc.returncode==0

def run_quality():
    data={'pylint':pylint_score(),'black_clean':black_check()}
    print(json.dumps(data,indent=2))
    return data
if __name__=='__main__':
    run_quality()
