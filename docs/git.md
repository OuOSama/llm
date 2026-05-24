# 🚀 Git Commit Message Style Guide

A clean and scalable commit message convention using Gitmoji to keep your Git history beautiful, readable, and professional ✨

---

# 📌 Format

```text
<emoji> <type>(<scope>): <short description>

[optional body]
```

---

# 🧩 Structure Breakdown

| Part                  | Description                                      |
| --------------------- | ------------------------------------------------ |
| `<emoji>`             | Gitmoji representing the type of change          |
| `<type>`              | Commit category such as `feat`, `fix`, or `docs` |
| `<scope>`             | Specific module, feature, or area affected       |
| `<short description>` | Short and clear explanation of the change        |
| `[optional body]`     | Additional details if needed                     |

---

# ✨ Recommended Rules

* Use **present tense** verbs such as `add`, `fix`, `update`
* Keep messages short and descriptive
* Avoid ending descriptions with periods
* Prefer lowercase writing
* Use `!` for **breaking changes**

---

# 📝 Real-World Examples

## ✨ Features & Logic

```text
✨ feat(auth): add google oauth2 login flow
✨ feat(node-editor): implement drag-and-drop for visual scripting nodes
💥 feat(api)!: switch primary database connection to supabase mode
```

---

## 🐛 Bug Fixes

```text
🐛 fix(db): resolve connection pool timeout on supabase production
🐛 fix(ui): fix memory leak when switching active tabs
✏️ fix(typos): correct misspelled variable names in engine controller
```

---

## ⚡ Performance & Architecture

```text
⚡ perf(render): optimize instanced static mesh rendering for grass
🏗️ refactor(backend): decouple command handler from node logic
♻️ refactor: clean up unused dependencies and optimize Cargo.toml
```

---

## 🗃️ Database & Configuration

```text
🗃️ db(migration): add trader_profile table and setup initial schema
📦 build(deps): upgrade sqlx to latest stable version
🔧 chore: add .env.example and improve gitignore rules
```

---

## 💄 UI, Visuals & Assets

```text
💄 style(theme): improve dark mode contrast and neon borders
🍱 assets: import new stylized 3D character models and textures
📝 docs: update installation guide and supabase setup instructions
```

---

# 🎯 Recommended Commit Types

| Type       | Purpose                            |
| ---------- | ---------------------------------- |
| `feat`     | New feature                        |
| `fix`      | Bug fix                            |
| `refactor` | Code restructuring                 |
| `perf`     | Performance improvements           |
| `docs`     | Documentation changes              |
| `style`    | UI or formatting updates           |
| `test`     | Tests and testing updates          |
| `build`    | Build system or dependency changes |
| `ci`       | CI/CD updates                      |
| `chore`    | Miscellaneous maintenance          |
| `assets`   | Images, models, sounds, resources  |
| `db`       | Database or migration changes      |

---

# 🌙 Example Commit Timeline

```text
✨ feat(auth): add jwt authentication middleware
🐛 fix(api): resolve panic on invalid refresh token
⚡ perf(cache): optimize redis session lookup
📝 docs(readme): update docker setup instructions
```

A clean Git history feels like a perfectly organized inventory UI in an RPG — readable, searchable, and satisfying to navigate ✨ค่ะ