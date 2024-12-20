import PropTypes from 'prop-types';
import merge from 'lodash/merge';
import ReactApexChart from 'react-apexcharts';
// @mui
import { Card, CardHeader, Box } from '@mui/material';
// components
import { BaseOptionChart } from '../../../components/chart';

// ----------------------------------------------------------------------

AppLatency.propTypes = {
  title: PropTypes.string,
  subheader: PropTypes.string,
  type: PropTypes.string,
  chartData: PropTypes.array.isRequired,
  chartLabels: PropTypes.arrayOf(PropTypes.string).isRequired,
  dashArrayValue: PropTypes.arrayOf(PropTypes.number)
};

export default function AppLatency({ title, subheader, chartLabels, chartData, dashArrayValue, type, ...other }) {

  const chartOptions = merge(BaseOptionChart(), {
    plotOptions: { bar: { columnWidth: '16%' } },
    fill: {
      type: chartData.map((i) => i.fill),
      colors: chartData.map((i) => i.color),
      opacity: 1,
    },
    // labels: chartLabels,
    xaxis: {
      type: 'datetime',
      categories: chartLabels, // Use categories for datetime type
      labels: {
        show: true,
        formatter: (value) => {
          const date = new Date(value);
          const day = String(date.getDate()).padStart(2, '0');
          const month = date.toLocaleString('default', { month: 'short' });
          return `${day} ${month}`;
        }
      }
    },
    yaxis: {
      min: 0,
      tickAmount: 6,
      forceNiceScale: true,
      title: {
        text: 'ms'
      },
      labels: {
        formatter: (y) => {
          if (typeof y !== 'undefined') {
            // return Math.pow(10, Math.ceil(Math.log10(v)));
            return type && y !== 0 ? `${(10 ** y).toFixed(0)}` : `${y.toFixed(0)}`;
          }
          return y;
        },
      }
    },
    stroke: {
      curve: 'straight',
      dashArray: dashArrayValue,
    },
    tooltip: {
      shared: true,
      intersect: false,
      y: {
        formatter: (y) => {
          if (y === 0)
            return `No data`;
          if (y === 1 && type === 'tail')
            return `No data`;

          if (typeof y !== 'undefined' && y !== 0) {
            // console.log(y);
            // return y
            return type ? `${(10 ** y).toFixed(0)} ms` : `${y.toFixed(0)} ms`;
          }

          return y;
        },
      },
    },
  });
  if (type === 'median') {
    chartOptions.yaxis.min = 1; // Replace with the minimum value you want
    chartOptions.yaxis.max = 4; // Replace with the maximum value you want
  }
  if (type === 'tail') {
    chartOptions.yaxis.min = 0; // Replace with the minimum value you want
    chartOptions.yaxis.max = 5; // Replace with the maximum value you want
  }

  return (
    <Card {...other} sx={{
      transition: "0.3s",
      margin: "auto",
      boxShadow: "0 8px 40px -12px rgba(0,0,0,0.2)",
      "&:hover": {
        boxShadow: "0 16px 70px -12.125px rgba(0,0,0,0.3)"
      },
    }}>
      <CardHeader title={title} subheader={subheader} />

      <Box sx={{ p: 3, pb: 1 }} dir="ltr">
        <ReactApexChart type="line" series={chartData} options={chartOptions} height={264} />
      </Box>
    </Card>
  );
}
