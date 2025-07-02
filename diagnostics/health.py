
import psutil, time, requests, json

def snapshot():
    return {'ts':time.time(),'cpu':psutil.cpu_percent(), 'mem':psutil.virtual_memory().percent,'disk':psutil.disk_usage('/').percent}

def latency(url='https://duckduckgo.com'):
    try:
        t0=time.time();requests.get(url,timeout=5);return (time.time()-t0)*1000
    except requests.RequestException:
        return -1

def run():
    data=snapshot();data['lat_ms']=latency();print(json.dumps(data,indent=2));return data
if __name__=='__main__':
    run()
