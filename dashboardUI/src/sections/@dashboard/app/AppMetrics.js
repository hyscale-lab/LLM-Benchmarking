import React from "react";
import PropTypes from "prop-types";
import merge from "lodash/merge";
import ReactApexChart from "react-apexcharts";
// @mui
import { Card, CardHeader, Box } from "@mui/material";
// Base chart options
import { BaseOptionChart } from "../../../components/chart";
import { providerColors } from "./AppMetricsDate";

// ----------------------------------------------------------------------

const AppMetrics = ({ title, metricType, subheader, metrics, min }) => {
    // Combine latencies and CDFs for all providers
    const chartData = Object.keys(metrics).map((provider) => {
        const providerData = metrics[provider];
        const modelKeys = Object.keys(providerData);

        return modelKeys.map((model) => {
            const metricData = providerData[model][metricType];
            if (!metricData) return null;

            const latencies = metricData.latencies.map(parseFloat);
            const cdf = metricData.cdf.map(parseFloat);
            console.log(provider, model)
            return {
                name: `${provider}`,
                type: "line", // Line with points
                data: latencies.map((latency, index) => ({
                    x: Math.log10(latency),
                    y: cdf[index],
                })),
                color: providerColors[provider] || "#999999",
            };
        }).filter(Boolean); // Remove null entries
    }).flat();
    const colors = chartData.map((entry) => entry.color);
    const allLatencies = chartData
        .flatMap(series => series.data.map(point => point.x))
        .filter(x => Number.isFinite(x));
    const maxLatency = Math.max(...allLatencies);
    const minLatency = Math.min(...allLatencies);
    const computedMin = (min != null && min > 0) ? Math.log10(min) : minLatency;
    // Chart options
    const chartOptions = merge(BaseOptionChart(), {
        stroke: {
            curve: "smooth",
            width: 2, // Ensure lines are prominent
        },
        colors,
        xaxis: {
            type: "numeric", // Logarithmic X-axis for latency
            title: {
                text: "Latency (ms)",
            },
            min: computedMin,
            max: maxLatency,
            labels: {
                formatter: (value) => `${(10 ** value).toFixed(3)} ms`, // Convert back to linear scale for display
            },
            tickAmount: 10,
        },
        yaxis: {
            title: {
                text: "Portion of Requests (CDF)",
            },
            labels: {
                formatter: (value) => value.toFixed(2), // Show intervals like 0.0, 0.2, ...
            },
            min: 0,
            max: 1,
            tickAmount: 5, // Ensure intervals are 0.2 each
        },
        tooltip: {
            shared: true, // Shared tooltip for better interactivity
            intersect: false,
            x: {
                formatter: (value) => `${(10 ** value).toFixed(3)} ms`,
            },
            y: {
                formatter: (value) => value.toFixed(3),
            },
        },
    });

    return (
        <Card
            sx={{
                transition: "0.3s",
                margin: "auto",
                boxShadow: "0 8px 40px -12px rgba(0,0,0,0.2)",
                "&:hover": {
                    boxShadow: "0 16px 70px -12.125px rgba(0,0,0,0.3)",
                },
            }}
        >
            <CardHeader title={title} subheader={subheader} />
            <Box sx={{ p: 3, pb: 1 }} dir="ltr">
                <ReactApexChart
                    type="line"
                    series={chartData} // Line chart with points
                    options={chartOptions}
                    height={450} // Increased height for better visualization
                />
            </Box>
        </Card>
    );
};

AppMetrics.propTypes = {
    title: PropTypes.string.isRequired, // Metric name (e.g., response_times, timetofirsttoken)
    metricType: PropTypes.string.isRequired,
    subheader: PropTypes.string, // Additional information to display
    metrics: PropTypes.object.isRequired, // Metrics object with provider -> model -> metric structure
    min: PropTypes.number,
};

export default AppMetrics;
