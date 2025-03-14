document.addEventListener("DOMContentLoaded", () => {
    const spinButton = document.getElementById("spin-button");
    const resultContainer = document.getElementById("result-container");
    const historyTable = document.getElementById("history-table-body");
    const historyContainer = document.getElementById("history-container");
    const historyToggle = document.getElementById("history-toggle");
    const columnColors = ["#FF5733", "#00F075", "#3357FF", "#FF33A1", "#FFD700", "#8A2BE2"];

    if (!spinButton || !resultContainer || !historyTable || !historyContainer) {
        console.error("❌ 필수 요소를 찾을 수 없음. HTML을 확인하세요!");
        return;
    }

    if (historyToggle) {
        historyToggle.addEventListener("click", () => {
            historyContainer.style.display = historyContainer.style.display === "none" ? "block" : "none";
        });
    }

    async function fetchPredictedNumbers() {
        try {
            const response = await fetch("http://18.224.64.89:5001/predict");
            if (!response.ok) {
                throw new Error(`❌ 서버 오류: ${response.status} ${response.statusText}`);
            }
            const data = await response.json();
            console.log("✅ API 응답:", data);
            
            if (!data["Optimal Games"] || !Array.isArray(data["Optimal Games"])) {
                throw new Error("잘못된 데이터 형식: " + JSON.stringify(data));
            }

            updateHistoryTable(data["History"] || []);
            return data["Optimal Games"]; 
        } catch (error) {
            console.error("❌ API 요청 실패:", error);
            alert("서버 오류! 랜덤 번호를 생성합니다.");
            return [];
        }
    }

    function updateHistoryTable(history) {
        console.log("📜 업데이트된 히스토리 데이터:", history);
        historyTable.innerHTML = "";
        history.forEach((games, index) => {
            let row = document.createElement("tr");
            
            let cell = document.createElement("td");
            cell.textContent = `${history.length - index}`;
            cell.classList.add("history-round");
            row.appendChild(cell);
            
            let numCell = document.createElement("td");
            numCell.innerHTML = games.map(game => `<span class="lotto-number">${game.join("</span> <span class='lotto-number'>")}</span>`).join(" | ");
            numCell.classList.add("history-numbers");
            row.appendChild(numCell);
            
            let copyButton = document.createElement("button");
            copyButton.innerHTML = "📋";
            copyButton.classList.add("copy-button");
            copyButton.onclick = () => copyToClipboard(games);
            let copyCell = document.createElement("td");
            copyCell.appendChild(copyButton);
            row.appendChild(copyCell);
            
            historyTable.appendChild(row);
        });
        historyContainer.style.display = "none";
    }
    
    function copyToClipboard(games) {
        const text = games.map(game => game.join(", ")).join(" | ");
        navigator.clipboard.writeText(text).then(() => {
            alert("✅ 복사 완료!");
        }).catch(err => {
            console.error("❌ 복사 실패:", err);
        });
    }
    
    async function startSlotAnimation() {
        console.log("🎰 'Spin' 버튼 클릭됨!");
        spinButton.disabled = true;
        resultContainer.innerHTML = "";

        const games = await fetchPredictedNumbers();
        if (Array.isArray(games) && games.length === 5) {
            games.forEach((game, gameIndex) => {
                let row = document.createElement("div");
                row.classList.add("result-row");

                game.forEach((num, index) => {
                    let slot = document.createElement("div");
                    slot.classList.add("slot");

                    let stopTime = 2000 + index * 500;  // ✅ 기존보다 대기 시간 증가 (기존 대비 +3초)
                    let interval = setInterval(() => {
                        slot.innerText = Math.floor(Math.random() * 45) + 1;
                    }, 50);  // ✅ 숫자 변경 속도를 기존 50ms → 100ms로 변경 (더 부드러운 애니메이션)

                    setTimeout(() => {
                        clearInterval(interval);
                        slot.innerText = num;
                        slot.style.backgroundColor = columnColors[index];
                    }, stopTime);

                    row.appendChild(slot);
                });

                resultContainer.appendChild(row);
            });
        } else {
            resultContainer.innerHTML = "❌ 로또 번호를 불러오는 데 실패했습니다.";
        }
        
        setTimeout(() => {
            spinButton.disabled = false;
        }, 7000);  // ✅ 전체 애니메이션 시간 증가에 맞춰 버튼 활성화 시간 조정
    }

    spinButton.addEventListener("click", startSlotAnimation);
    console.log("✅ 이벤트 리스너 등록 완료!");
    
    // ✅ 히스토리 테이블 스타일 조정
    historyContainer.style.backgroundColor = "black";
    historyContainer.style.color = "black";
    historyContainer.style.padding = "10px";

    // ✅ 히스토리 토글 버튼 스타일 적용
    const historyToggleContainer = document.getElementById("history-toggle-container");
    if (historyToggleContainer) {
        historyToggleContainer.style.marginTop = "20px";
    }

    if (historyToggle) {
        historyToggle.style.border = "1px solid white";
        historyToggle.style.background = "none";
        historyToggle.style.color = "white";
        historyToggle.style.padding = "8px 16px";
    }

    // ✅ 히스토리 테이블 스타일 적용
    const historyTableElement = document.getElementById("history-table");
    if (historyTableElement) {
        historyTableElement.style.borderCollapse = "collapse";
        historyTableElement.style.width = "80%";
        historyTableElement.style.margin = "0 auto";
    }

    document.querySelectorAll("#history-table th").forEach(th => {
        th.style.backgroundColor = "white";
        th.style.color = "black";
    });

    document.querySelectorAll("#history-table td").forEach(td => {
        td.style.backgroundColor = "black";
        td.style.color = "white";
    });
});

