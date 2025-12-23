# Payment Screening System

A comprehensive payment screening and compliance system that performs sanctions checks, watchlist matching, and risk assessment for financial transactions. This project includes both a web UI and a REST API implementation.

## Overview

This system screens payment transactions against:
- **Watchlists**: UN Sanctions, OFAC SDN, EU Consolidated lists
- **Sanctioned Countries**: Iran, Syria, Ukraine, Pakistan, Cuba, South Korea
- **Name Matching**: Using Jaro-Winkler and Jaccard similarity algorithms
- **Address Verification**: Detection of sanctioned country mentions in addresses

## Project Structure

```
PaymentScreening/
├── paymentscreening_v3.py          # Flask Web UI Application
├── payment_screening_api.py         # Flask REST API
├── test_payloads.json              # Test scenarios documentation
├── test_payloads_flask_api/        # Individual test payloads for API testing
│   ├── payload_scenario_1_high_risk_match.json
│   ├── payload_scenario_2_sanctioned_country.json
│   ├── payload_scenario_3_watchlist_name_match.json
│   ├── payload_scenario_4_low_risk.json
│   ├── payload_scenario_5_pakistan_address.json
│   ├── payload_scenario_6_china_watchlist.json
│   ├── payload_scenario_7_minimal_match.json
│   └── payload_scenario_8_iran_sanctioned.json
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 1. Flask Web UI Application

**File**: `paymentscreening_v3.py`

### Description

A full-featured web application with an interactive UI for screening payment transactions. Includes a floating AI-powered explanation panel that provides audit-ready compliance reports.

### Features

- ✅ **Interactive Web Form**: Input payer and beneficiary details
- ✅ **Real-time Screening**: Instant risk assessment and decision
- ✅ **Visual Dashboard**: Color-coded decision display (ESCALATE/RELEASE)
- ✅ **Detailed Breakdown**: View all candidate matches with scores
- ✅ **AI Explain Panel**: On-premise explanation generator for compliance
- ✅ **Responsive Design**: Bootstrap 5 UI with dark theme

### Risk Scoring Algorithm

The system uses a composite risk score calculated as:

```
Score = 0.60 × Name_Similarity + 0.30 × Address_Similarity + 0.05 × DOB_Match + Country_Bonus
```

Where:
- **Name Similarity**: Max of Jaro-Winkler and Jaccard similarity across primary name and AKAs
- **Address Similarity**: 0.4 × Jaro-Winkler + 0.6 × Jaccard on tokenized addresses
- **DOB Match**: 1.0 if exact match, 0.0 otherwise
- **Country Bonus**: 0.05 if country codes match exactly

### Decision Logic

- **ESCALATE (Sanctioned Country)**: If party country is sanctioned OR address mentions sanctioned country
- **ESCALATE (Score Threshold)**: If composite score ≥ 0.80
- **RELEASE**: If score < 0.80 and no sanction hits

### How to Run

```bash
# Install dependencies
pip install flask

# Run the application
python paymentscreening_v3.py

# Access in browser
http://127.0.0.1:5092
```

### Usage

1. Open `http://127.0.0.1:5092` in your browser
2. Fill in payer and beneficiary details
3. Click **"Screen Payment"**
4. Review the decision and detailed breakdown
5. Click **"AI Explain"** for compliance report (optional)

### Default Test Data

The form loads with pre-populated test data:
- **Payer**: Global Trade LLC (matches watchlist)
- **Beneficiary**: Olena Petrenko-Kovalenko in Ukraine (sanctioned country)
- **Result**: ESCALATE due to sanctioned country

### AI Explain Panel

The floating panel generates audit-ready compliance explanations including:
- Decision summary and reasoning
- Watchlist context and category
- Key risk drivers (name, address, DOB, country)
- Sanctions hit details
- Recommended actions for Level-2 review

---

## 2. Flask REST API

**File**: `payment_screening_api.py`

### Description

A lightweight REST API for programmatic access to the payment screening system. Designed for integration with other systems, automated workflows, and testing with tools like Postman.

### API Endpoints

#### 1. **GET /** - API Documentation
Returns API information and available endpoints.

```bash
GET http://127.0.0.1:5000/
```

**Response:**
```json
{
  "message": "Payment Screening API",
  "version": "1.0",
  "endpoints": { ... }
}
```

#### 2. **POST /screen** - Screen Payment Transaction
Main endpoint for screening payments.

```bash
POST http://127.0.0.1:5000/screen
Content-Type: application/json
```

**Request Body:**
```json
{
  "payer_name": "Global Trade LLC",
  "payer_address": "PO Box 12345, Dubai, UAE",
  "payer_country": "AE",
  "payer_dob": "",
  "benef_name": "John Smith",
  "benef_address": "123 Main Street, London, UK",
  "benef_country": "GB",
  "benef_dob": "",
  "amount": 12500.00,
  "currency": "USD",
  "reference": "Invoice 2025-10-ACME"
}
```

**Response:**
```json
{
  "timestamp": "2025-12-23T10:30:00",
  "screening_result": {
    "decision": "ESCALATE",
    "reason": "Score Threshold",
    "best_role": "PAYER",
    "best_score": 0.950,
    "breakdown": { ... },
    "sanction_flag": false,
    "candidates": [ ... ]
  },
  "transaction_details": {
    "amount": 12500.00,
    "currency": "USD",
    "reference": "Invoice 2025-10-ACME"
  }
}
```

#### 3. **GET /health** - Health Check
Check API availability.

```bash
GET http://127.0.0.1:5000/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-23T10:30:00"
}
```

#### 4. **GET /watchlist** - Get Watchlist
Retrieve all watchlist entries.

```bash
GET http://127.0.0.1:5000/watchlist
```

#### 5. **GET /sanctioned-countries** - Get Sanctioned Countries
Retrieve sanctioned countries list.

```bash
GET http://127.0.0.1:5000/sanctioned-countries
```

### How to Run

```bash
# Install dependencies
pip install flask

# Run the API
python payment_screening_api.py

# API will be available at
http://127.0.0.1:5000
```

### Testing with Postman

1. **Start the API:**
   ```bash
   python payment_screening_api.py
   ```

2. **In Postman:**
   - Create a new POST request
   - URL: `http://127.0.0.1:5000/screen`
   - Headers: `Content-Type: application/json`
   - Body: Select "raw" and "JSON"
   - Copy any payload from `test_payloads_flask_api/` folder
   - Click **Send**

3. **Test Scenarios Available:**
   - **Scenario 1**: High risk watchlist match → ESCALATE
   - **Scenario 2**: Sanctioned country (Ukraine) → ESCALATE
   - **Scenario 3**: Name matches Mohammad Al Hamed → ESCALATE
   - **Scenario 4**: Low risk transaction → RELEASE
   - **Scenario 5**: Pakistan address mention → ESCALATE
   - **Scenario 6**: Zhang Wei watchlist match → ESCALATE
   - **Scenario 7**: Minimal similarity → RELEASE
   - **Scenario 8**: Iran sanctioned country → ESCALATE

### Error Handling

The API returns appropriate HTTP status codes:
- **200**: Successful screening
- **400**: Bad request (missing fields, invalid data)
- **404**: Endpoint not found
- **405**: Method not allowed
- **500**: Internal server error

---

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Install Dependencies

```bash
pip install flask
```

Or using requirements.txt:

```bash
pip install -r requirements.txt
```

---

## Watchlist Data

The system includes dummy watchlist entries for demonstration:

| Name | List | Category | Country | DOB |
|------|------|----------|---------|-----|
| Mohammad Al Hamed | UN Sanctions | Terrorism | BH | 1978-04-09 |
| Zhang Wei | OFAC SDN | Proliferation | CN | 1983-11-23 |
| Hafiz Mohammed | EU Consolidated | Corruption | PK | 1990-02-01 |
| Global Trade LLC | Internal Watch | Adverse Media | AE | N/A |

### Sanctioned Countries

- Pakistan (PK)
- Iran (IR)
- Syria (SY)
- Ukraine (UA)
- Cuba (CU)
- South Korea (KR)

**Note**: Aliases like "ukraise" are mapped to "ukraine"

---

## Technical Details

### Algorithms Used

1. **Jaro-Winkler Distance**: String similarity for name/address matching
2. **Jaccard Similarity**: Token-based similarity for addresses
3. **Text Normalization**: Lowercase, punctuation removal, abbreviation expansion
4. **Tokenization**: Space-based splitting for comparison

### Abbreviation Expansion

Common abbreviations are automatically expanded:
- st/str → street
- rd → road
- ave/av → avenue
- blvd → boulevard
- ln → lane
- p.o. box → po box

---

## Use Cases

### Financial Institutions
- Pre-transaction screening for wire transfers
- AML/CFT compliance automation
- Real-time sanctions checking

### Compliance Teams
- Batch screening of payment queues
- Investigation of flagged transactions
- Audit trail generation

### Integration Teams
- API integration with payment gateways
- ERP system plugins
- Automated workflow triggers

---

## Security Considerations

⚠️ **This is a demonstration system with dummy data.** 

For production use:
- Use official OFAC, UN, EU watchlists via APIs
- Implement authentication and authorization
- Add rate limiting and request validation
- Encrypt sensitive data in transit and at rest
- Implement comprehensive audit logging
- Use environment variables for configuration
- Deploy behind a firewall/VPN

---

## Future Enhancements

- [ ] Integration with live sanctions list APIs
- [ ] Machine learning for name matching
- [ ] Multi-language support
- [ ] Database backend for audit logs
- [ ] User authentication and role-based access
- [ ] Batch processing for multiple transactions
- [ ] Advanced reporting and analytics
- [ ] Webhook notifications for escalations
- [ ] Docker containerization
- [ ] OpenAPI/Swagger documentation

---

## License

This project is for educational and demonstration purposes.

---

## Support

For questions or issues, please refer to the inline code documentation or create an issue in the repository.

---

**Last Updated**: December 23, 2025
