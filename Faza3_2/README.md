# Fáza 3.2 – Franka FR3 a DQN

V tejto časti som riešil, aby robot dosiahol zadaný bod presnejšie a zvládal ciele v rôznych častiach pracovnej oblasti. Pracoval som s modelom Franka FR3 v MuJoCo. Cieľ sa zadáva súradnicami X, Y, Z v metroch.

## Ako to funguje

Použil som Double DQN spolu s plánovačom. Plánovač najprv pomocou inverznej kinematiky vypočíta polohu kĺbov pre daný cieľ a cestu rozdelí na menšie kroky. DQN potom rozhoduje, ktorým kĺbom a ktorým smerom robot pohne. Má 15 možností: pohyb každého zo siedmich kĺbov v oboch smeroch alebo držanie polohy. Pred vykonaním príkazu sa ešte kontroluje možná kolízia.

DQN teda riadi pohyb medzi medzipolohami, ale celú cestu neplánuje sám. V kóde je aj RRT-Connect pre prípad, keď priama cesta nevyhovuje. Pri uložených 216 testoch však nebol potrebný.

Pri učení som použil ukážky pohybov zo simulácie a potom tréning Double DQN. Zo 160 epizód sa získalo 6744 prechodov. Sieť má 12 vstupných príznakov na akciu a dve skryté vrstvy po 64 neurónov. Pri spustení už vyberá akcie naučená sieť, bez simulovaného učiteľa.

## Čo sa podarilo

Robot musí dostať koncový bod ramena do vzdialenosti najviac **1 mm od cieľa**. Zároveň sa musí spomaliť pod stanovenú hranicu 5 mm/s a zostať v tejto tolerancii aspoň 0,3 s. Na jednu úlohu má 30 s simulácie.

Otestoval som 216 úloh: 72 cieľov z domácej polohy, 72 pri hraniciach oblasti a 72 s rôznymi štartovacími polohami. Tieto úlohy sa nepoužili na výber modelu.

| Riešenie | Dosiahnuté ciele | Priemerná konečná chyba | Kolízie |
|---|---:|---:|---:|
| Predchádzajúci lokálny DQN | 100/216 | 319,30 mm | 10 |
| Plánovač + Double DQN | 215/216 | 0,22 mm | 0 |

Úspešnosť nového riešenia bola **99,54 %**. Jedna úloha sa nestihla dokončiť v limite a skončila s chybou 1,051 mm. Aj tento neúspech je započítaný vo výsledkoch. Porovnávam tu celé riešenia, takže zlepšenie nie je iba zásluhou samotného DQN.

Ciele sú rozložené do 12 smerových sektorov a do rôznych výšok a vzdialeností. Test nepokrýva každý možný bod, ale dáva lepšiu predstavu o správaní robota mimo okolia domácej polohy. Koncový bod zostáva aspoň 3 cm nad podlahou; to je obmedzenie pracovnej oblasti, nie tolerancia cieľa. Podrobnosti výberu úloh sú v [BENCHMARK.md](results/workspace/BENCHMARK.md).

![Výsledky testovania](results/workspace/visualization/workspace_coverage.png)

## Ukážky pohybu

[Animácia robota](results/workspace/visualization/robot_replay.gif) ukazuje zrýchlené prehratie jednej zaznamenanej úspešnej úlohy. Zelený bod je pre lepšiu viditeľnosť zväčšený a neukazuje veľkosť tolerancie.

V [interaktívnej vizualizácii](results/workspace/visualization/workspace_viewer.html) sú všetky testované ciele a 23 vybraných trajektórií vrátane neúspešnej. Dá sa meniť pohľad a prehrávať pohyb. HTML treba po stiahnutí otvoriť v prehliadači, priamo na GitHube sa zobrazuje len jeho kód.

## Spustenie

Použil som Linux, Python 3.12, MuJoCo 3.10 a PyTorch 2.13. GPU nie je potrebné. Príkazy sa spúšťajú z hlavného priečinka repozitára:

```bash
python3 -m venv Faza3_2/.venv
Faza3_2/.venv/bin/python -m pip install -r Faza3_2/requirements.txt
bash Faza3_2/run_workspace_demo.sh
```

Natrénovaný model je už uložený. Demo zobrazí 12 vybraných úspešných úloh; nejde o celý test. Medzi úlohami sa simulácia resetuje na štartovaciu polohu. Vlastný cieľ sa dá zadať takto:

```bash
Faza3_2/.venv/bin/python Faza3_2/code/run_workspace_dqn.py \
  --goal -0.45 0.25 0.55 --viewer
```

Na opakovanie všetkých 216 úloh bez okna:

```bash
OMP_NUM_THREADS=1 Faza3_2/.venv/bin/python Faza3_2/code/run_workspace_dqn.py \
  --tasks Faza3_2/results/workspace/test_tasks.json \
  --output Faza3_2/results/workspace/hybrid_test_repeat.json
```

Výstup musí mať nový názov, aby sa neprepísali uložené výsledky. Zdrojový kód je v `code/`, merania a model v `results/workspace/`. Skript `train_joint_waypoint_dqn.py` slúži na nové učenie. Automatické kontroly sa spúšťajú príkazom:

```bash
OMP_NUM_THREADS=1 Faza3_2/.venv/bin/python -m unittest discover -s Faza3_2/tests -v
```

## Čo ešte zostáva

Zatiaľ som riešenie overil iba v simulácii. Robot dosahuje polohu XYZ, ale nerieši požadovanú orientáciu uchopovača. Testy neobsahovali nové vonkajšie prekážky, záťaž nástroja ani šum a oneskorenie snímačov. Používa sa ideálna kompenzácia gravitácie. Výsledok 1 mm preto zatiaľ nemôžem preniesť ako tvrdenie o presnosti reálneho robota. Ďalším krokom by bolo doplniť pripojenie k robotu, kalibráciu a overenie pohybových limitov.

Model FR3 pochádza z MuJoCo Menagerie / Franka Description. [Pôvodná licencia](mujoco/franka_fr3/LICENSE) a [opis úprav modelu](mujoco/franka_fr3/PROJECT_MODIFICATIONS.md) sú priložené.
