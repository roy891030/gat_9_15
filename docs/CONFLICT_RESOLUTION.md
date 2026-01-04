# 合併衝突處理指南

本指南整理在多人協作或切換版本時常見的合併衝突解法，並搭配本專案的目錄與腳本命名，方便快速對齊。

## 1. 快速檢查當前狀態
1. 確認目前分支與是否有未完成的合併：
   ```bash
   git status
   ```
2. 若看到 `You have unmerged paths` 或 `both modified`，代表存在衝突檔案。

> 已知常見衝突檔案：`PROJECT_STRUCTURE_SUMMARY.md`、`README.md`、`docs/PROJECT_OVERVIEW.md`、`docs/RUNPODS_GUIDE.md`、`requirements.txt`、`visualize_factor_attention.py`。

## 2. 若合併不想要，先中止
如果誤觸合併或狀態混亂，可先中止並回到乾淨工作樹：
```bash
git merge --abort   # 若在 merge 狀態
```
或直接捨棄暫存修改：
```bash
git reset --hard
```

## 3. 手動解衝突的步驟
1. 打開衝突檔案，搜尋標記 `<<<<<<<`、`=======`、`>>>>>>>`。
2. 依需求選擇保留「當前分支」或「遠端分支」內容，或手動合併兩者，移除所有標記。
3. 針對本專案常見檔案的建議：
   - **專案文件（`PROJECT_STRUCTURE_SUMMARY.md`、`README.md`、`docs/*.md`）**：保留最新的專案結構與執行指令表，避免重複段落或舊路徑。
   - **`requirements.txt`**：合併兩邊的依賴並去重，保持依賴按字母排序。
   - **`visualize_factor_attention.py`**：確認 CLI 參數（尤其是 `--weights`、`--artifact_dir`）一致，避免遺漏新的旗標。

## 4. 完成後的確認
1. 檔案移除衝突標記後，加入暫存：
   ```bash
   git add <file1> <file2> ...
   ```
2. 確認沒有未處理的衝突：
   ```bash
   git status
   ```
   工作樹應顯示乾淨或只剩下已暫存的修改。
3. 可選：驗證 Python 腳本能成功編譯，確保合併不破壞語法：
   ```bash
   python -m py_compile visualize_factor_attention.py train_baselines.py train_dmfm_wei2022.py train_gat_fixed.py
   ```
4. 提交：
   ```bash
   git commit -m "Resolve merge conflicts"
   ```

## 5. 若仍有遠端更新
合併完成後，如需再拉取最新上游，可使用 rebase 方式減少新的衝突：
```bash
git pull --rebase
```
若出現新的衝突，重複上面的第 3–4 步。

## 6. 復原已提交的錯誤合併（進階）
若已推送錯誤合併，可透過 revert 回復：
```bash
git revert <bad_merge_commit_sha>
```
或建立新分支重新整理，將乾淨的檔案 cherry-pick 過來。

---

以上步驟可搭配 `docs/FUNCTIONAL_COMMAND_MAP.md` 與 README 的指令表，快速檢查合併後的腳本路徑、參數與輸出位置，確保專案可直接執行。
