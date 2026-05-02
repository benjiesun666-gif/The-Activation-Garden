import json, os, re

base = 'D:/pythonstudy/python_task/eml/experiments'
with open('D:/pythonstudy/python_task/eml/paper_draft.md', 'r', encoding='utf-8') as f:
    c = f.read()

print('=' * 60)
print('  NMI FORMAT COMPLIANCE CHECK')
print('=' * 60)

# Abstract words
abs_start = c.find('## Abstract')
abs_end = c.find('---', abs_start)
abs_text = c[abs_start+11:abs_end]
abs_words = len([w for w in abs_text.split() if w.strip()])
ok = abs_words <= 150
print('Abstract: {} words (limit 150) -> {}'.format(abs_words, 'PASS' if ok else 'FAIL'))

# Body structure
meth_start = c.find('## Methods')
body = c[abs_end:meth_start]
body_clean = re.sub(r'\$\$.*?\$\$', '', body, flags=re.DOTALL)
body_clean = re.sub(r'\$.*?\$', '', body_clean)
body_clean = re.sub(r'[#\*\|\[\]\(\)\-\+\{\}\\]', ' ', body_clean)
body_words = len([w for w in body_clean.split() if w.strip()])
ok = body_words <= 3500
print('Body text: ~{} words (limit 3500) -> {}'.format(body_words, 'PASS' if ok else 'FAIL'))

# Structure
print('Structure:')
print('  Results section: {}'.format('PASS' if 'What We Observed' in c else 'FAIL'))
print('  Discussion: {}'.format('PASS' if '## Discussion' in c else 'FAIL'))
print('  Methods: {}'.format('PASS' if '## Methods' in c else 'FAIL'))

# References
ref_section = c[c.find('## References'):]
ref_count = len(re.findall(r'^\d+\.\s', ref_section, re.M))
years = re.findall(r'\((\d{4})\)', ref_section)
recent = sum(1 for y in years if int(y) >= 2021)
ok_r = ref_count >= 25
ok_p = 100 * recent >= 50 * ref_count
print('References: {} total (min 25) -> {}'.format(ref_count, 'PASS' if ok_r else 'FAIL'))
print('  From 2021+: {}/{} ({}%) -> {}'.format(recent, ref_count, 100*recent//ref_count, 'PASS' if ok_p else 'FAIL'))

# Tables
tables = len(re.findall(r'(?i)\*Table\s+\d', c[:meth_start]))
ok = tables <= 6
print('Tables: {} (limit 6) -> {}'.format(tables, 'PASS' if ok else 'FAIL'))

# Data files
print('\nData files:')
files = ['mnist/results_mnist.json', 'cifar10/results_cifar10.json',
         'cifar10/results_gated_eml.json', 'cifar10/results_gated_resnet.json',
         'pinn/pinn_results.json', 'feynman/feynman_results.json',
         'deep_convergence/results_deep.json']
for fn in files:
    fp = os.path.join(base, fn)
    ok = os.path.exists(fp)
    print('  {} {}'.format('OK' if ok else 'FAIL', fn))

print('\n=== ALL CHECKS COMPLETE ===')
