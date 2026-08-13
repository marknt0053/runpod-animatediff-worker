#!/bin/bash
# strongブランチにpushしてRunPodのServerlessページを自動で開く
git add -A
git commit -m "${1:-update}"
git push origin strong
echo ""
echo "✅ Push完了"
echo "🔗 RunPodのServerlessページを開いています..."
echo "📌 cz74u1nurpeke6 (runpod-animatediff-worker) のManage→Edit Endpointを操作してください"
open "https://www.runpod.io/console/serverless"
