# Sherman: SolidWorks parts → STEP (one page) / המרת חלקי SolidWorks ל‑STEP (עמוד אחד)

2026-10-28. For: the CAD person / Shaked. Any PC with SolidWorks works. Nothing gets installed.
עבור: איש ה‑CAD / שקד. מתאים לכל מחשב עם SolidWorks. לא מתקינים שום דבר.

## 1. What to run / מה להריץ

**English.** Use the macro `SldprtToStep.bas`, sent together with this sheet.

Setup, once:
1. In SolidWorks: **Tools → Macro → New**. Save as `SldprtToStep.swp`.
2. In the editor that opens: **File → Import File** → `SldprtToStep.bas`.
3. Right-click the empty module `SldprtToStep1` → **Remove** → **No**.
4. **File → Save**, then close the editor.

Every run:
1. **Tools → Macro → Run** → `SldprtToStep.swp`.
2. Pick the folder with the parts.
3. Pick a **new, empty** output folder.
4. Wait for the "Done" box.

The macro opens each part **read-only** and never saves it. It exports STEP AP214 in mm and puts your
STEP setting back afterwards.

**No macro?** Use **SOLIDWORKS Task Scheduler → Export Files**:
- file type: **Part (*.sldprt)**, with subfolders;
- export to: **STEP AP214**;
- output: a new empty folder;
- **Run Now**.

Do **not** use "Convert Files": it re-saves the parts.

**עברית.** משתמשים במאקרו `SldprtToStep.bas`, שמצורף לדף הזה.

הגדרה, פעם אחת:
1. ב‑SolidWorks: **Tools → Macro → New**. שומרים בשם `SldprtToStep.swp`.
2. בעורך שנפתח: **File → Import File** ← `SldprtToStep.bas`.
3. קליק ימני על המודול הריק `SldprtToStep1` ← **Remove** ← **No**.
4. **File → Save**, וסוגרים את העורך.

בכל הרצה:
1. **Tools → Macro → Run** ← `SldprtToStep.swp`.
2. בוחרים את תיקיית החלקים.
3. בוחרים תיקיית פלט **חדשה וריקה**.
4. מחכים להודעת "Done".

המאקרו פותח כל חלק **לקריאה בלבד** ולעולם לא שומר אותו. הוא מייצא STEP AP214 במ"מ ומחזיר אחר כך את הגדרת ה‑STEP שלכם.

**בלי מאקרו?** משתמשים ב‑**SOLIDWORKS Task Scheduler → Export Files**:
- סוג קובץ: **Part (*.sldprt)**, כולל תת‑תיקיות;
- ייצוא ל‑**STEP AP214**;
- פלט: תיקייה חדשה וריקה;
- **Run Now**.

**לא** להשתמש ב‑"Convert Files": הוא שומר מחדש את החלקים.

## 2. Which parts first / אילו חלקים קודם

**English.** Start with these 84 part numbers: we already have a physical scan for each of them.
- Keep the original file names. Do not rename.
- If a part has several configurations, export the one used in production (the active one).

**עברית.** מתחילים מ‑84 מספרי החלקים האלה: לכל אחד מהם כבר יש לנו סריקה פיזית.
- משאירים את שמות הקבצים המקוריים, בלי לשנות שם.
- אם לחלק יש כמה קונפיגורציות, מייצאים את זו שבייצור (הפעילה).

10835000_01 12851000_B 21763000_E 22352000_01 23976000_C 26687004_A01 26687021_A01 27796002_D 27796005_C
30820000_A01 34371016_A 36192000_02 37071000_A 38169000_04 38539000_02 38965002_A 40851000_C 41017000_B
42426000_01 42725000_B 42821000_C 43161000_A 43244000_B 43780000_B 43798000_A 43895000_B 43920000_A
43976000_A 44001000_B 44556000_A 45352000_B 46045000_D 46414000_C 47040000_C 47247004_03 47251002_02
47480000_C 47555020_1 47632001_A 48099000_01 48333018_00 48340007_01 48412000_B 48565005_A 48565022_A
48758000_A 48910000_A 48932000_B 48933000_B 48934000_B 48938000_C 49025000_A 49125000_A00 49162000_A
49163000_A 49186000_A 49187000_B 49188000_A 49191030_01 49191037_01 49191038_02 49191054_02 49227000_A
49291000_A 49292000_A 49293001_B 49355000_A 49438000_A 49461000_A 49553000_A 49565000_A 49566000_A
50278001_A 50482000_A 50535000_04 50537000_04 50539000_05 50552000_04 50553000_04 50566000_04 50567000_05
23317000_C1 37126000_D 37361005_01

**English.** All other parts are welcome later.

**עברית.** את כל שאר החלקים אפשר לשלוח אחר כך.

## 3. What to send back / מה לשלוח חזרה

**English.**
1. Zip the **whole output folder**: the `.step` files **and** `manifest.csv`.
2. Send the zip to us through your usual channel.

Do not send the SLDPRT files. Failed parts are listed in `manifest.csv` with the reason; nothing else is needed.

**עברית.**
1. מכווצים ל‑ZIP את **כל תיקיית הפלט**: קבצי `.step` **וגם** `manifest.csv`.
2. שולחים לנו את ה‑ZIP בערוץ הרגיל.

לא צריך לשלוח קבצי SLDPRT. חלקים שנכשלו מופיעים ב‑`manifest.csv` עם הסיבה, ולא צריך שום דבר נוסף.

---
**On our side.** We validate each STEP and import it as a **draft** part, never released: `python -m apps.production_line.recipes.cad_import`. Technical details are in `research/zeroshot/zs_rust/station/sldprt_to_step/README.md`. Full list with evidence: `INVENTORY.md`.
