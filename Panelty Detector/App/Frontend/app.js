document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        const targetTab = this.dataset.tab;
        
        // Remove active class from all tabs and content
        document.querySelectorAll('.tab-btn').forEach(tab => tab.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
        
        // Add active class to clicked tab and corresponding content
        this.classList.add('active');
        document.getElementById(`${targetTab}-tab`).classList.add('active');
    });
});

// Analysis function for both tabs
async function analyzeScenario(scenario, resultsDiv, fineType = 'onspot') {
    const loadingDiv = fineType === 'court' ? document.getElementById('courtLoading') : document.getElementById('loading');
    const progressBar = loadingDiv.querySelector('.loading-progress-bar');
    
    resultsDiv.innerHTML = '';

    if (!scenario) {
        resultsDiv.innerHTML = '<div style="color:#c62828;font-weight:500;padding:20px;text-align:center;background:rgba(198,40,40,0.1);border-radius:12px;">Please describe the traffic offence scenario.</div>';
        return;
    }

    // Show loading animation
    loadingDiv.classList.add('show');
    progressBar.classList.add('animate');

    try {
        const response = await fetch('http://127.0.0.1:5000/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scenario, fineType })
        });

        if (!response.ok) {
            throw new Error('API request failed');
        }

        const data = await response.json();

        // Hide loading animation
        loadingDiv.classList.remove('show');
        progressBar.classList.remove('animate');

        if (!Array.isArray(data) || data.length === 0) {
            resultsDiv.innerHTML = fineType === 'court' 
                ? '<div class="no-court-results">No court fine information found for the given scenario.</div>'
                : '<div style="color:#757575;padding:20px;text-align:center;background:rgba(255,255,255,0.05);border-radius:12px;">No offences detected for the given scenario.</div>';
            return;
        }

        // Handle court fine display as list
        if (fineType === 'court') {
            let listHTML = '<div class="court-results-list">';
            
            data.forEach(row => {
                listHTML += `
                    <div class="court-fine-item">
                        <div class="offence-title">${row.Offence}</div>
                        
                        <!-- Add description section -->
                        <div class="offence-description">
                            <div class="description-label">Description:</div>
                            <div class="description-content">${row.Description}</div>
                        </div>
                        
                        <!-- Section Details Header -->
                        <div class="section-header">
                            <div class="section-header-title">Legal Reference</div>
                            <div class="section-details">
                `;
                
                const sectionDetails = row.SectionDetails || {};
                
                if (sectionDetails.SectionNumber && sectionDetails.SectionNumber !== '-') {
                    listHTML += `
                        <div class="section-detail">
                            <div class="section-detail-label">Section</div>
                            <div class="section-detail-value">${sectionDetails.SectionNumber}</div>
                        </div>
                    `;
                }
                
                if (sectionDetails.Subsection && sectionDetails.Subsection !== '-') {
                    listHTML += `
                        <div class="section-detail">
                            <div class="section-detail-label">Subsection</div>
                            <div class="section-detail-value">${sectionDetails.Subsection}</div>
                        </div>
                    `;
                }
                
                if (sectionDetails.Paragraph && sectionDetails.Paragraph !== '-') {
                    listHTML += `
                        <div class="section-detail">
                            <div class="section-detail-label">Paragraph</div>
                            <div class="section-detail-value">${sectionDetails.Paragraph}</div>
                        </div>
                    `;
                }
                
                if (sectionDetails.Subparagraph && sectionDetails.Subparagraph !== '-') {
                    listHTML += `
                        <div class="section-detail">
                            <div class="section-detail-label">Subparagraph</div>
                            <div class="section-detail-value">${sectionDetails.Subparagraph}</div>
                        </div>
                    `;
                }
                
                listHTML += `
                            </div>
                        </div>
                        
                        <!-- Penalties Section -->
                        <div class="penalties-section">
                            <div class="penalties-title">
                                <i class="fas fa-gavel"></i> Court Penalties
                            </div>
                `;
                
                if (row.Penalties && row.Penalties.length > 0) {
                    row.Penalties.forEach((penalty, index) => {
                        listHTML += `
                            <div class="penalty-item">
                                <div class="conviction-type">
                                    <i class="fas fa-balance-scale"></i> ${penalty.conviction_type}
                                </div>
                                <div class="fine-details">
                                    <div class="fine-amount">
                                        <i class="fas fa-money-bill-wave"></i> ${penalty.penalty_details.fine_amount}
                                    </div>
                        `;
                        
                        // Add additional penalties if they exist
                        if (penalty.penalty_details.additional_penalties && penalty.penalty_details.additional_penalties.length > 0) {
                            listHTML += `
                                <div class="additional-penalties">
                            `;
                            penalty.penalty_details.additional_penalties.forEach(additionalPenalty => {
                                const icon = additionalPenalty.toLowerCase().includes('suspension') ? 'fas fa-ban' : 
                                           additionalPenalty.toLowerCase().includes('imprisonment') ? 'fas fa-lock' : 'fas fa-exclamation-triangle';
                                
                                listHTML += `
                                    <div class="additional-penalty-item">
                                        <i class="${icon}"></i> ${additionalPenalty}
                                    </div>
                                `;
                            });
                            listHTML += `
                                </div>
                            `;
                        }
                        
                        listHTML += `
                                </div>
                            </div>
                        `;
                        
                        // Add separator between penalty items (except for the last one)
                        if (index < row.Penalties.length - 1) {
                            listHTML += `<div class="penalty-separator"></div>`;
                        }
                    });
                } else {
                    listHTML += `
                        <div class="penalty-item">
                            <div class="fine-amount">
                                <i class="fas fa-info-circle"></i> Court fine information not available
                            </div>
                        </div>
                    `;
                }
                
                listHTML += `
                        </div>
                    </div>
                `;
            });
            
            listHTML += '</div>';
            resultsDiv.innerHTML = listHTML;
            return;
        }

        // Handle on-spot fine display as table (existing logic)
        // Calculate total fine (extract numbers from fine strings)
        let totalFine = 0;
        let hasFineRange = false;
        
        data.forEach(row => {
            // Extract numeric value from fine string
            const fineStr = String(row.Fine);
            
            // Check if it's a range (e.g., "Rs. 2,500 - Rs. 5,000")
            if (fineStr.includes(' - ')) {
                hasFineRange = true;
                // For ranges, take the minimum value for total calculation
                const match = fineStr.split(' - ')[0].replace(/,/g, '').match(/(\d+)/g);
                if (match) {
                    totalFine += match.map(Number).reduce((a, b) => a + b, 0);
                }
            } else {
                // Single value fine
                const match = fineStr.replace(/,/g, '').match(/(\d+)/g);
                if (match) {
                    totalFine += match.map(Number).reduce((a, b) => a + b, 0);
                }
            }
        });

        // Build table for on-spot fines
        let tableHTML = `
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Traffic Offence</th>
                        <th>On-Spot Fine Amount</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        data.forEach(row => {
            tableHTML += `
                <tr>
                    <td>${row.Offence}</td>
                    <td><span class="fine-amount">${row.Fine}</span></td>
                </tr>
            `;
        });
        
        // Add total row for on-spot fines
        if (totalFine > 0) {
            const totalText = hasFineRange ? 'Total Minimum Fine' : 'Total Fine Amount';
            tableHTML += `
                <tr class="total-row">
                    <td><strong>${totalText}</strong></td>
                    <td><strong class="fine-amount">Rs. ${totalFine.toLocaleString()}</strong></td>
                </tr>
            `;
        }
        
        tableHTML += `
                </tbody>
            </table>
        `;
        resultsDiv.innerHTML = tableHTML;
        
    } catch (err) {
        // Hide loading animation on error
        loadingDiv.classList.remove('show');
        progressBar.classList.remove('animate');
        
        resultsDiv.innerHTML = `<div style="color:#b91c1c;padding:20px;text-align:center;background:rgba(185,28,28,0.1);border-radius:12px;border:1px solid rgba(185,28,28,0.3);">
            <i class="fas fa-exclamation-triangle" style="margin-right:8px;"></i>
            Error: ${err.message}
        </div>`;
    }
}

// On Spot Fine Analysis
document.getElementById('analyzeBtn').addEventListener('click', async function() {
    const scenario = document.getElementById('scenarioInput').value.trim();
    const resultsDiv = document.getElementById('results');
    await analyzeScenario(scenario, resultsDiv, 'onspot');
});

// Court Fine Analysis
document.getElementById('courtAnalyzeBtn').addEventListener('click', async function() {
    const scenario = document.getElementById('courtScenarioInput').value.trim();
    const resultsDiv = document.getElementById('courtResults');
    await analyzeScenario(scenario, resultsDiv, 'court');
});
