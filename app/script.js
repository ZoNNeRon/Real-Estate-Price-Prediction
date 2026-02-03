// ===== GLOBAL STATE =====
let currentSection = 'dashboard';
let currentMarket = 'ames';
let currentCalculator = 'ames-calc';

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', () => {
    initializeNavigation();
    initializeMarketSwitching();
    initializeCalculators();
});

// ===== NAVIGATION =====
function initializeNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    
    navButtons.forEach(button => {
        button.addEventListener('click', () => {
            const section = button.dataset.section;
            switchSection(section);
            
            // Update active state
            navButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
        });
    });
}

function switchSection(sectionName) {
    currentSection = sectionName;
    
    // Hide all sections
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });
    
    // Show selected section
    document.getElementById(sectionName).classList.add('active');
}

// ===== MARKET SWITCHING =====
function initializeMarketSwitching() {
    const marketButtons = document.querySelectorAll('.market-btn');
    
    marketButtons.forEach(button => {
        button.addEventListener('click', () => {
            const market = button.dataset.market;
            
            // Handle dashboard market switching
            if (market === 'ames' || market === 'malaysia') {
                switchDashboardMarket(market);
                
                // Update active state for dashboard buttons
                const dashboardSection = document.querySelector('#dashboard');
                dashboardSection.querySelectorAll('.market-btn').forEach(btn => {
                    btn.classList.remove('active');
                });
                button.classList.add('active');
            }
            
            // Handle calculator market switching
            if (market === 'ames-calc' || market === 'malaysia-calc') {
                switchCalculatorMarket(market);
                
                // Update active state for calculator buttons
                const calculatorSection = document.querySelector('#calculator');
                calculatorSection.querySelectorAll('.market-btn').forEach(btn => {
                    btn.classList.remove('active');
                });
                button.classList.add('active');
            }
        });
    });
}

function switchDashboardMarket(market) {
    currentMarket = market;
    
    // Hide all market dashboards
    document.querySelectorAll('.market-dashboard').forEach(dashboard => {
        dashboard.classList.remove('active');
    });
    
    // Show selected dashboard
    document.getElementById(`${market}-dashboard`).classList.add('active');
}

function switchCalculatorMarket(market) {
    currentCalculator = market;
    
    // Hide all calculator forms
    document.querySelectorAll('.calculator-form').forEach(form => {
        form.classList.remove('active');
    });
    
    // Show selected calculator
    document.getElementById(market).classList.add('active');
    
    // Hide any previous results
    document.querySelectorAll('.result-box').forEach(box => {
        box.style.display = 'none';
    });
}

// ===== CALCULATOR FUNCTIONALITY =====
function initializeCalculators() {
    // Ames Calculator
    const amesForm = document.getElementById('ames-form');
    amesForm.addEventListener('submit', (e) => {
        e.preventDefault();
        calculateAmesPrice();
    });
    
    // Malaysia Calculator
    const malaysiaForm = document.getElementById('malaysia-form');
    malaysiaForm.addEventListener('submit', (e) => {
        e.preventDefault();
        calculateMalaysiaPrice();
    });
}

// ===== AMES PRICE CALCULATION =====
function calculateAmesPrice() {
    // Get form values
    const quality = parseInt(document.getElementById('ames-quality').value);
    const area = parseInt(document.getElementById('ames-area').value);
    const year = parseInt(document.getElementById('ames-year').value);
    const garage = parseInt(document.getElementById('ames-garage').value);
    const basement = parseInt(document.getElementById('ames-basement').value);
    const bathrooms = parseInt(document.getElementById('ames-bathrooms').value);
    
    // Simplified price calculation based on feature importance
    // This is a simplified model approximation for demonstration
    // In production, this would call an actual ML model API
    
    // Base price
    let basePrice = 100000;
    
    // Quality impact (most important feature - ~65% importance)
    basePrice += quality * 22000;
    
    // Living area impact
    basePrice += area * 65;
    
    // Year built (newer = higher price)
    const age = 2010 - year;
    basePrice -= age * 1200;
    
    // Garage capacity
    basePrice += garage * 8000;
    
    // Basement area
    basePrice += basement * 35;
    
    // Bathrooms
    basePrice += bathrooms * 5000;
    
    // Add some randomness to simulate model variance
    const variance = (Math.random() - 0.5) * 10000;
    let estimatedPrice = basePrice + variance;
    
    // Ensure price is within reasonable bounds
    estimatedPrice = Math.max(50000, Math.min(800000, estimatedPrice));
    
    // Calculate confidence interval (±RMSE)
    const rmse = 18632;
    const minPrice = Math.max(0, estimatedPrice - rmse);
    const maxPrice = estimatedPrice + rmse;
    
    // Display result
    displayAmesResult(estimatedPrice, minPrice, maxPrice);
}

function displayAmesResult(price, min, max) {
    const resultBox = document.getElementById('ames-result');
    const priceElement = document.getElementById('ames-price');
    const minElement = document.getElementById('ames-min');
    const maxElement = document.getElementById('ames-max');
    
    // Format numbers with commas
    priceElement.textContent = Math.round(price).toLocaleString();
    minElement.textContent = Math.round(min).toLocaleString();
    maxElement.textContent = Math.round(max).toLocaleString();
    
    // Show result with animation
    resultBox.style.display = 'block';
    resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    
    // Add pulse animation
    resultBox.classList.add('loading');
    setTimeout(() => {
        resultBox.classList.remove('loading');
    }, 1000);
}

// ===== MALAYSIA PRICE CALCULATION =====
function calculateMalaysiaPrice() {
    // Get form values
    const size = parseInt(document.getElementById('my-size').value);
    const psf = parseInt(document.getElementById('my-psf').value);
    const location = document.getElementById('my-location').value;
    const rooms = parseInt(document.getElementById('my-rooms').value);
    const bathrooms = parseInt(document.getElementById('my-bathrooms').value);
    const parking = parseInt(document.getElementById('my-parking').value);
    
    // Price calculation heavily influenced by PSF (77% importance)
    let basePrice = size * psf;
    
    // Location premium
    const locationMultipliers = {
        'klcc': 1.4,
        'bukit_bintang': 1.3,
        'mont_kiara': 1.2,
        'damansara': 1.1,
        'cheras': 0.9,
        'other': 1.0
    };
    basePrice *= locationMultipliers[location];
    
    // Rooms adjustment
    basePrice += rooms * 15000;
    
    // Bathrooms adjustment
    basePrice += bathrooms * 10000;
    
    // Parking adjustment
    basePrice += parking * 25000;
    
    // Add variance
    const variance = (Math.random() - 0.5) * 50000;
    let estimatedPrice = basePrice + variance;
    
    // Ensure price is within reasonable bounds
    estimatedPrice = Math.max(100000, Math.min(5000000, estimatedPrice));
    
    // Calculate confidence interval (±RMSE)
    const rmse = 156786;
    const minPrice = Math.max(0, estimatedPrice - rmse);
    const maxPrice = estimatedPrice + rmse;
    
    // Display result
    displayMalaysiaResult(estimatedPrice, minPrice, maxPrice);
}

function displayMalaysiaResult(price, min, max) {
    const resultBox = document.getElementById('malaysia-result');
    const priceElement = document.getElementById('malaysia-price');
    const minElement = document.getElementById('malaysia-min');
    const maxElement = document.getElementById('malaysia-max');
    
    // Format numbers with commas
    priceElement.textContent = Math.round(price).toLocaleString();
    minElement.textContent = Math.round(min).toLocaleString();
    maxElement.textContent = Math.round(max).toLocaleString();
    
    // Show result with animation
    resultBox.style.display = 'block';
    resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    
    // Add pulse animation
    resultBox.classList.add('loading');
    setTimeout(() => {
        resultBox.classList.remove('loading');
    }, 1000);
}

// ===== UTILITY FUNCTIONS =====
function formatCurrency(value, currency = 'USD') {
    if (currency === 'USD') {
        return `$${Math.round(value).toLocaleString()}`;
    } else if (currency === 'MYR') {
        return `RM ${Math.round(value).toLocaleString()}`;
    }
    return Math.round(value).toLocaleString();
}

function formatNumber(value) {
    return Math.round(value).toLocaleString();
}

// ===== ERROR HANDLING =====
window.addEventListener('error', (e) => {
    console.error('Application error:', e);
});

// ===== EXPORT FOR TESTING =====
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        calculateAmesPrice,
        calculateMalaysiaPrice,
        formatCurrency,
        formatNumber
    };
}