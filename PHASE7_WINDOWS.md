# Faza 7 — Integrimi dhe verifikimi në Windows/PyCharm

Baza: `feat/random-search`, commit `355adee126cff1ab9c4d2ece368e25042b542b7f`.
Puna e re: `work/advanced-optimization`. Nuk është bërë push ose Pull Request.

## 1. Nxirre ZIP-in në një dosje të veçantë

Shembujt më poshtë supozojnë këtë strukturë pas **Extract All**:

```text
C:\Users\vesa-\Downloads\Interferometric_Mission_Faza_7\
    project\tools\integrate_phase7.py
    git\phase7.bundle
```

Nëse e nxjerr diku tjetër, ndrysho vetëm rrugët e paketës në komandat më poshtë.
Mos kopjo dosjen `project` mbi worktree-n ekzistues. Integrimi kryhet nga Git bundle.

## 2. Kontrollo paraprakisht integrimin

Në terminalin PowerShell të PyCharm, me Python-in ekzistues:

```powershell
python "C:\Users\vesa-\Downloads\Interferometric_Mission_Faza_7\project\tools\integrate_phase7.py" --repo "C:\Users\vesa-\PycharmProjects\Interferometric-Mission-Development" --bundle "C:\Users\vesa-\Downloads\Interferometric_Mission_Faza_7\git\phase7.bundle" --worktree "C:\Users\vesa-\PycharmProjects\Interferometric-Mission-Phase7" --check-only
```

Ky kontroll verifikon bundle-in dhe prejardhjen nga baza; importon objektet Git,
por nuk krijon branch/worktree dhe nuk ndryshon skedarët e punës.
Nëse baza mungon, kryej fillimisht `git fetch origin feat/random-search` nga
repository yt dhe përsërite kontrollin. Nëse dosja ose branch-i i destinacionit
ekziston me përmbajtje tjetër, skripti ndalet; nuk e mbishkruan.

## 3. Krijo worktree-n dhe branch-in e fazës 7

Ekzekuto të njëjtën komandë pa `--check-only`:

```powershell
python "C:\Users\vesa-\Downloads\Interferometric_Mission_Faza_7\project\tools\integrate_phase7.py" --repo "C:\Users\vesa-\PycharmProjects\Interferometric-Mission-Development" --bundle "C:\Users\vesa-\Downloads\Interferometric_Mission_Faza_7\git\phase7.bundle" --worktree "C:\Users\vesa-\PycharmProjects\Interferometric-Mission-Phase7"
```

Krijohet branch-i lokal `feat/advanced-optimization`. Dosja e re është **linked
Git worktree i të njëjtit repository**, jo repository i rindërtuar. Branch-i
`feat/random-search`, `main`, commit-et e vjetra, staging area dhe skedarët
e worktree-t të vjetër ruhen. Nuk përdoren `reset`, `clean`, `stash` ose push.
Edhe skedarët untracked/ignored të worktree-t të vjetër mbeten në vend.

Hape dosjen `Interferometric-Mission-Phase7` në PyCharm dhe zgjidh interpreterin
ekzistues Conda `interferometric-mission`. Nuk nevojitet një environment i dytë.

```powershell
cd "C:\Users\vesa-\PycharmProjects\Interferometric-Mission-Phase7"
git branch --show-current
git log --oneline -8
git status --short
python main.py --check-environment
```

Nëse environment-i nuk është aktiv, aktivizoje me `conda activate
interferometric-mission`. Nëse mungojnë varësi, përdor:

```powershell
conda env update -n interferometric-mission -f environment.yml
```

Ruhet `heyoka.py=3.2.0`, sepse evaluatori i pandryshuar ESA përdor API-n e atij
versioni. Faza 7 nuk shton varësi shkencore.

## 4. Testet dhe tri zgjidhjet e ruajtura

```powershell
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
python -m unittest discover -s tests -v
python main.py --instance beginner --evaluate --chromosome evidence/phase7/best/beginner.best.json
python main.py --instance intermediate --evaluate --chromosome evidence/phase7/best/intermediate.best.json
python main.py --instance hard --evaluate --chromosome evidence/phase7/best/hard.best.json
```

Rezultatet e pritshme lexohen te `evidence/phase7/best-index.json` dhe raporti.
CLI ekzistues duhet të shfaqë `Direct ESA agreement: True`. Testet e paketës
janë ekzekutuar në Linux; këto komanda janë verifikimi që duhet kryer në Windows.

## 5. Ekzekutime të reja

Fillimisht provo një buxhet të vogël:

```powershell
python -m src.advanced_experiments --instances intermediate hard --variants stm-random stm-ga stm-de sage stm-local --budget 1500 --seeds 11 --output results/phase7-local-smoke
```

Përsërit krahasimin formal të paketës, me seed-et që nuk u përdorën gjatë pilotit:

```powershell
python -m src.advanced_experiments --instances beginner intermediate hard --variants random-search ga de stm-random stm-ga stm-de sage stm-local --budget 10000 --seeds 101 202 303 --stage confirmation --output results/phase7-reproduction/confirmation
```

Ky krahasim kryen 72 ekzekutime: **720,000 vlerësime kërkimi dhe 72 kontrolle
zyrtare shtesë**. Koha në Ryzen 5 mund të ndryshojë nga serveri Linux. Mund ta
ndajësh sipas instancës, duke përdorur dosje të ndryshme rezultatesh.

`stm-local` është heqja e fazës gjenetike nga SAGE; `sage` përdor 20% të buxhetit
për GA dhe 80% për local search. `stm-random` izolon efektin e inicializimit.
`stm-ga-blend`, `stm-de-fixed` dhe `stm-mixed` ruhen për riprodhimin e ablacioneve
të pilotit. Algoritmet bazë `ga`, `de`, `random-search` janë të pandryshuara.

Mos e ripërdor të njëjtën dosje output: programi ndalet nëse ajo tashmë përmban
një eksperiment. Çdo `.best.json` është vektor i sheshtë, në renditjen zyrtare
`x[1..N], y[1..N], z[1..N], vx[1..N], vy[1..N], vz[1..N]`.

Për studimin e inicializimit dhe grafikët nga ekzekutimet e reja:

```powershell
python -m src.initialization_study --output results/phase7-reproduction/initialization --samples 100 --seeds 901 902 903
python -m src.phase7_reporting --root results/phase7-reproduction
```

Komanda e raportimit kryen **gjashtë kontrolle fitness shtesë**, dy për secilën
zgjidhje përfundimtare. Grafikët rigjenerohen nga rezultatet e ruajtura; nuk
ekzekutohet optimizim i fshehur.

## 6. Nëse ke vazhduar punën pas `355adee`

Skripti nuk bashkon automatikisht commit-et e tua më të reja me fazën 7.
Ato mbeten në branch-in e tyre. Worktree i fazës 7 nis nga baza e kërkuar dhe
ruan historinë e saj; një bashkim i mëvonshëm bëhet pasi të shqyrtohen ndryshimet.
Për t'u kthyer te puna e mëparshme, hape dosjen e vjetër në PyCharm.
