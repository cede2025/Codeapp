#!/data/data/com.termux/files/usr/bin/bash

echo "📁 [1/5] Sprawdzanie istnienia katalogu Codeapp/"
if [ ! -d "Codeapp" ]; then
  echo "❌ Folder Codeapp nie istnieje w bieżącym katalogu. Przerywam."
  exit 1
fi

echo "🧹 [2/5] Usuwanie .git z folderu Codeapp/"
rm -rf Codeapp/.git

echo "📦 [3/5] Usuwanie wpisu Codeapp z indeksu Git"
git rm --cached -r Codeapp

echo "➕ [4/5] Ponowne dodanie folderu Codeapp jako zwykłego katalogu"
git add Codeapp
git commit -m "Fix: dodano Codeapp jako zwykły folder (usunięto subrepo)"

echo "📤 [5/5] Wysyłanie zmian do GitHub (main)"
git push origin main

echo "✅ Gotowe. Folder Codeapp jest teraz zwykłym katalogiem w repozytorium."
