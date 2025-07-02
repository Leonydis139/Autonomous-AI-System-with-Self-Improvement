
import subprocess, pathlib, json, sys, tempfile, os
ROOT = pathlib.Path(__file__).resolve().parents[1]
SELF_PATH = pathlib.Path(__file__).resolve()

BLACK_ARGS = ['black', '--quiet']
PYLINT_ARGS = ['pylint', str(ROOT), '-f', 'json']


def pylint_score():
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
        subprocess.run(PYLINT_ARGS + ['-o', tmp.name], check=False)
        data = json.load(open(tmp.name))
    os.unlink(tmp.name)
    if not data:
        return 10.0
    return sum(m.get('score', 0) for m in data) / len(data)


def auto_format():
    subprocess.run(BLACK_ARGS + [str(ROOT)], check=False)


def main(threshold=8.5):
    before = pylint_score()
    print('Pylint score before', before)
    if before >= threshold:
        print('Quality satisfactory.')
        return
    print('Running black auto-formatting...')
    auto_format()
    after = pylint_score()
    print('Pylint score after', after)

if __name__ == '__main__':
    main()
