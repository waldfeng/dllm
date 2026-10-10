"""Retrieve primary USPTO publications and record actual access results."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import json
import pymupdf

ROOT = Path('/tmp/opencode')
IDS = ['20190266236', '20220157045', '20250342340', '20220358358',
       '20260148045', '20200279150', '20190251423', '20250292084',
       '20260252911', '20260039311', '20250356166']

def fetch(number):
    url = f'https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/{number}'
    result = subprocess.run(['curl','-L','-sS','--connect-timeout','8',
                             '--max-time','40',url],capture_output=True)
    record = {'number':number,'url':url,'bytes':len(result.stdout)}
    if result.stdout.startswith(b'%PDF'):
        path = ROOT / f'US{number}A1.pdf'
        path.write_bytes(result.stdout)
        doc = pymupdf.open(path)
        text = '\n'.join(page.get_text() for page in doc)
        (ROOT / f'US{number}A1.txt').write_text(text)
        record.update(status='PDF obtained',pages=len(doc),text_characters=len(text))
        print(number, record, '\n', text[-18000:], flush=True)
    else:
        record.update(status='failed',error=result.stderr.decode(errors='replace'))
        print(record,flush=True)
    return record

if __name__ == '__main__':
    records = []
    with ThreadPoolExecutor(8) as pool:
        for future in as_completed([pool.submit(fetch,n) for n in IDS]):
            records.append(future.result())
    (ROOT / 'patent_access_v14.json').write_text(json.dumps(records,indent=2))
