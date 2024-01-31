## 1. Approach and Methodology

### FastAPI Selection

- FastAPI Python web server framework was chosen to implement this RESTful service for two reasons:

  - **FastAPI Saves Time**
    - With an out-of-the-box web-framework like FastAPI, one can get set up within minutes with minimal boilerplate.
    - This framework took only 5 minutes to set up (by copy-pasting the code from the landing page), allowing the focus to be on writing the logic for each endpoint. Testing and debugging are also faster as FastAPI automatically generates swagger documentation for each endpoint, allowing developers to check the docs endpoint to see what format the parameters to each endpoint need to take.
  - **FastAPI is Industrial Strength and Can Scale**
    - This framework provides all the necessary middleware for dealing with CORS, authentication/authorization, dependency injection, testing, validation, and documentation. This is a massive time-saver and allows the project to quickly scale with minimal extra testing/risk for writing faulty code.

- **Walking through the different features added to the service**

  - **Endpoints**

    - `/` base endpoint: this should function as the health check endpoint for other services to monitor the availability of this endpoint
    - `tickers` - returns a list of tickers that the endpoint supports
    - `/returns` - takes in three path parameters `ticker`, `start_date`, and `end_date` and returns a JSON object with the time series of daily returns
    - `/correlation` - takes in 4 path parameters `ticker1`, `ticker2`, `start_date`, and `end_date` and returns the correlation between the daily returns time series following the same logic as the returns endpoint. The only difference is that the pandas `.corr()` function with the default 'Pearson' correlation is used to calculate the correlation between both series and this information is returned as a JSON object
    - `/correlation_matrix` - handles POST requests and extracts a `CorrelationMatrixParams` object from the body. Return series are calculated for each ticker, concatenated, and `corr()` is called once again to get a correlation matrix

  - **Helper Functions**

    - Since the endpoints share logic, three helper functions were created:
      - `price_data`: fetches price data
      - `returns`: fetches daily returns
      - `correlations`: constructs a correlation matrix from each of the provided tickers in the post body

  - **Documentation**

    - To view the automatically generated open API docs, visit `http://<YOUR_URL>:8000/docs` if running locally, `YOUR_URL` is just `localhost` otherwise it's just the docs endpoint for whatever base URL the service is hosted at

  - **Input Validation**

    - Ensuring users do not pass in malformed or potentially malicious input is crucial in API design. In addition, validations allow avoiding writing redundant validation checks for each endpoint, keeping the code clean and reusable. A simple ticker validation was provided that checks whether the ticker parameter is supported by the endpoint. The same logic could be extended to check for valid date ranges.

  - **Caching**

    - Cache decorators were added to the `price_data` and `returns` functions to improve speed on the user side and avoid unnecessary calls to the Yahoo Finance endpoint. Of course, caching comes with an increased memory footprint so in a real-world setting, the cache strategy needs to be tailored to the expected usage/resource constraints.

  - **Middleware**

    - Currently, CORS middleware is set to the default configurations, thus this endpoint would be publicly available and insecure if hosted online. These were provided as a reminder that the middleware and authentication should be implemented before production.

  - **Testing**

    - Unit tests were written for each endpoint to illustrate how testing can be done with this framework. It uses the pytest framework and is very straightforward.

  - **Error Handling**
    - Logic was added to explicitly handle exceptions using FastAPI's built HTTPException class. This follows best practice of returning precise and useful HTTPExceptions that can help users understand what's going on with the server. For example, if the input validation check fails, the service will return a 422 (Unprocessable Entity) exception as opposed to some generic error which lets the user know that the parameter was understood but the server doesn't support it.

## 2. Challenges Faced and Future Considerations

- Despite using a production-grade web-server framework with all the bells and whistles, the challenge with an investment data service like this primarily centered around dealing with potentially unreliable third-party dependencies such as yfinance.
- More specifically, the most common challenges real-world market data API's include:

  1. Maintaining awareness and tolerance to changes in underlying data (stock splits, mergers, name changes, etc.)

  - Addressing this usually involves maintaining a suite of tests which can preemptively ascertain when certain tickers are no longer valid or the scale/format of the data has changed due to splits.
  - In addition, many vendors maintain endpoints specifically for monitoring corporate actions which can also be used to build tolerance for these changes.

  2. Making logic robust to missing or erroneous data

  - Missing data can be dealt with by adding checks for completeness (ensuring that every date for which the security was traded has a price).
  - Erroneous data is harder to catch, but one potential solution is to add some checks that might watch for large inconsistencies/discrepancies in the series. If discrepancies are detected, warnings could be issued to the end user to double-check their data.
