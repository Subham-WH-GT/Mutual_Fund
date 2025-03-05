const fetch = require('node-fetch');
 
const url = 'https://api.ipoalerts.in/ipos?status=open';
const options = {
  method: 'GET',
  headers: {
    'x-api-key': 'a518696c60a470172ff2f77c0ab8746ea559d720f8cb59ed6c907020c671eea8'
  }
};
 
fetch(url, options)
  .then(response => response.json())
  .then(data => console.log(data))
  .catch(error => console.error('Error fetching IPO data:', error));