"""Syntax + import check only — no training."""
import py_compile, os, sys, traceback

base = 'D:/pythonstudy/python_task/eml/experiments'
files = []

for root, dirs, fnames in os.walk(base):
    dirs[:] = [d for d in dirs if d not in ['__pycache__','figures','archive']]
    for fn in fnames:
        if fn.endswith('.py'):
            files.append(os.path.join(root, fn))

ok = 0
failed = []
for fp in sorted(files):
    rel = fp.replace(base+'/', '')
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            code = f.read()
        # Syntax check
        compile(code, fp, 'exec')
        ok += 1
        print('  [OK] %s' % rel)
    except SyntaxError as e:
        failed.append((rel, 'SyntaxError: %s' % e))
        print('  [SYNTAX ERROR] %s: %s' % (rel, e))
    except Exception as e:
        failed.append((rel, 'Read error: %s' % e))
        print('  [ERROR] %s: %s' % (rel, e))

print('\nResults: %d/%d passed' % (ok, len(files)))
if failed:
    for rel, err in failed:
        print('  FAIL: %s -> %s' % (rel, err))
