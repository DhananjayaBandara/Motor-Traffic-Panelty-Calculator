document.getElementById('analyzeBtn').addEventListener('click', async function() {
    const scenario = document.getElementById('scenarioInput').value.trim();
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';

    if (!scenario) {
        resultsDiv.innerHTML = '<div style="color:#c62828;font-weight:500;">Please describe the traffic offence scenario.</div>';
        return;
    }

    resultsDiv.innerHTML = '<span style="color:#2563eb;font-weight:500;">Analyzing...</span>';

    try {
        const response = await fetch('http://127.0.0.1:5000/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scenario })
        });

        if (!response.ok) {
            throw new Error('API request failed');
        }

        const data = await response.json();

        if (!Array.isArray(data) || data.length === 0) {
            resultsDiv.innerHTML = '<div style="color:#757575;">No offences detected for the given scenario.</div>';
            return;
        }

        // Calculate total fine (extract numbers from fine strings)
        let totalFine = 0;
        data.forEach(row => {
            // Extract numeric value from fine string (e.g., "Rs. 2,500")
            const match = String(row.Fine).replace(/,/g, '').match(/(\d+)/g);
            if (match) {
                totalFine += match.map(Number).reduce((a, b) => a + b, 0);
            }
        });

        // Build table (Offence and Fine columns only)
        let tableHTML = `
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Offence</th>
                        <th>Fine</th>
                    </tr>
                </thead>
                <tbody>
        `;
        data.forEach(row => {
            tableHTML += `
                <tr>
                    <td>${row.Offence}</td>
                    <td>${row.Fine}</td>
                </tr>
            `;
        });
        // Add total row
        tableHTML += `
                <tr>
                    <td style="font-weight:700;color:#1a237e;">Total</td>
                    <td style="font-weight:700;color:#1a237e;">Rs. ${totalFine.toLocaleString()}</td>
                </tr>
        `;
        tableHTML += `
                </tbody>
            </table>
        `;
        resultsDiv.innerHTML = tableHTML;
    } catch (err) {
        resultsDiv.innerHTML = `<span style="color:#b91c1c;">Error: ${err.message}</span>`;
    }
});
