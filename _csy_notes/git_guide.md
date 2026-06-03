# 🤖 开源项目 Fork & 学习笔记管理标准工作流

本工作流适用于：**Fork 开源项目进行源码阅读、在原代码中写注释、以及新建专属文件夹记录 Jupyter Notebook 笔记**。它可以确保你的本地修改与原作者的更新完美隔离，永远不丢失代码。

---

git config --global user.email "你的GitHub绑定邮箱@example.com"

git config --global user.name "你的GitHub用户名"

git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890





---

## 阶段一：云端准备与本地克隆（一次性配置）

当发现一个想上手的开源项目时，先执行以下步骤：

1. **网页端 Fork**：
   在 GitHub 打开目标开源项目页面，点击右上角的 **Fork** 按钮，将其复制到你自己的账号下。
2. **克隆到本地**：
   打开本地终端（Terminal），将**你自己的 Fork 仓库**克隆到电脑上：
   ```bash
   git clone https://github.com
   ```
3. **关联原作者仓库（建立上游通道）**：
   进入项目文件夹，把原作者的仓库添加为远程上游（命名为 `upstream`），用于日后拉取更新：
   ```bash
   cd 项目名
   git remote add upstream https://github.com
   ```
   * *提示：可用 `git remote -v` 检查是否成功。你应该能看到 `origin`（指向你自己）和 `upstream`（指向原作者）两个地址。*

---

## 阶段二：建立独立学习区（规范化管理）

克隆完成后，**立刻建立隔离的开发环境**，避免直接在主分支（`main` 或 `master`）上修改。

1. **创建并切换到专属学习分支**：
   ```bash
   git checkout -b study-notes
   ```
2. **新建专属笔记文件夹**：
   在项目根目录下，新建一个原项目**压根不存在**的文件夹（例如 `_my_notes/`）。
   * 💡 **防冲突核心原理**：在这个专属文件夹里新建你的 Jupyter Notebook (`.ipynb`)。因为原作者的仓库没有这个文件夹，所以后续无论原项目怎么更新，你的笔记文件**百分之百不会产生 Git 代码冲突**。

---

## 阶段三：日常学习与日常同步（循环使用）

### 🔄 场景 A：今天学完了，想把代码注释和 Notebook 笔记保存到 GitHub 云端
在本地写完注释、更新完 Notebook 后，在终端执行**经典三连命令**：
```bash
git add .
git commit -m "update: 记录今天的学习心得"
git push -u origin study-notes
```
* *注意：只有第一次推送新分支时需要加 `-u origin study-notes`。以后在该分支下更新，直接输入简短的 `git push` 即可。*
* 

### 🔄 场景 B：原作者更新了代码，我想把新代码合并到我的学习分支
当原项目发布了新版本或修复了 Bug，在本地按照以下 4 步进行安全合并：

1. **获取原作者的最新代码**：
   ```bash
   git fetch upstream
   ```
2. **切换回主分支，并将主分支与原作者同步对齐**：
   ```bash
   git checkout main
   git merge upstream/main
   ```
3. **切换回你的学习分支**：
   ```bash
   git checkout study-notes
   ```
4. **把对齐后的主分支代码，合并到你的学习分支中**：
   ```bash
   git merge main
   ```

#### ⚠️ 场景 B 的代码冲突（Conflict）处理指南：
* **如果你修改了原项目的核心代码**（比如在某行代码后面写了注释），而原作者恰好也修改了该行，第 4 步会提示冲突。
* **解决办法**：用 VS Code 打开冲突的文件，VS Code 会高亮显示两者的差异。由于你只是写了注释，直接在代码上方点击 **「保留双方更改 (Accept Both Changes)」**，然后重新 `git add .`、`git commit` 即可。
