import React, { useEffect, useState, useCallback } from "react";
import PropTypes from "prop-types";
import axios from "axios";
import { Grid, Container, Typography, CircularProgress, Alert, Stack } from "@mui/material";
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import Select from '@mui/material/Select';
import Page from "../../../components/Page";
import AppMetrics from "./AppMetrics";
import AppMetricsDate from "./AppMetricsDate";

// ----------------------------------------------------------------------

const AppMetricsPage = ({ metricType, streaming = true, title = "Metrics Dashboard", metricName, min, cdf = false, showInputType = true }) => {
    const [metrics, setMetrics] = useState(null);
    const [periodMetrics, setPeriodMetrics] = useState(null);
    const [dateList, setDateList] = useState(null);
    const [loadingMetrics, setLoadingMetrics] = useState(true);
    const [loadingPeriodMetrics, setLoadingPeriodMetrics] = useState(true);
    const [error, setError] = useState(false);
    const [inputType, setInputType] = useState("static");
    const [dateRange, setDateRange] = useState("three-month");
    const [selectedDate, setSelectedDate] = useState(null); // Initially null to ensure correct fetch order

    const baseURL = process.env.REACT_APP_BASE_URL;

    const fetchPeriodMetrics = useCallback(async () => {
        setLoadingPeriodMetrics(true);
        try {
            const response = await axios.get(`${baseURL}/metrics/period`, {
                params: { timeRange: dateRange, metricType, streaming, inputType },
            });
            setPeriodMetrics(response.data.aggregated_metrics);
            setDateList(response.data.date_array);

            // If dateList has values, set selectedDate to the first element
            if (response.data.date_array.length > 0) {
                setSelectedDate(response.data.date_array[0]);
            }
        } catch (error) {
            console.error("Error fetching period metrics:", error);
            setError(true);
        } finally {
            setLoadingPeriodMetrics(false);
        }
    }, [baseURL, dateRange, metricType, streaming, inputType]);

    const fetchMetrics = useCallback(async () => {
        if (!selectedDate) return; // Ensure selectedDate is set before fetching metrics
        setLoadingMetrics(true);
        try {
            const response = await axios.get(`${baseURL}/metrics/date`, {
                params: { date: selectedDate, metricType, streaming, inputType },
            });
            setMetrics(response.data.metrics);
        } catch (error) {
            console.error("Error fetching metrics:", error);
            setError(true);
        } finally {
            setLoadingMetrics(false);
        }
    }, [baseURL, selectedDate, metricType, streaming, inputType]);

    useEffect(() => {
        fetchPeriodMetrics();
    }, [fetchPeriodMetrics]);

    useEffect(() => {
        if (selectedDate && cdf) {
            fetchMetrics();
        }
    }, [selectedDate, fetchMetrics, cdf]);

    const handleInputTypeChange = (event) => {
        setInputType(event.target.value)
    }

    const handleDateRangeChange = (event) => {
        setDateRange(event.target.value);
    };

    const handleDateChange = (event) => {
        setSelectedDate(event.target.value);
    };

    // Check for error first to prevent getting stuck in loading state
    if (error)
        return (
            <Alert variant="outlined" severity="error">
                Something went wrong while fetching metrics!
            </Alert>
        );
    // Then check for loading
    // Combine loading states
    const loading = (loadingMetrics && cdf) || loadingPeriodMetrics;
    if (loading) return <CircularProgress />;

    if ((!metrics && cdf) || !periodMetrics) return <Typography>No data available</Typography>;

    const yaxis = metricType === "aime_2024_accuracy" ? "Accuracy" : "Latency";
    return (
        <Page title="Metrics Dashboard">
            <Container maxWidth="xl">
                <Typography variant="h4" sx={{ mb: 2 }}>
                    {title}
                </Typography>

                <Grid container spacing={3}>
                    {/* Stack to hold dropdowns */}
                    <Stack 
                        direction="row" 
                        alignItems="center" 
                        spacing={3}
                        sx={{ mb: 2, pt: 3, pl: 3 }}
                    >
                        {/* First Dropdown: Time Span */}
                        <Stack direction="row" alignItems="center">
                            <InputLabel sx={{ mr: 3 }}>Time Span:</InputLabel>
                            <Select
                                value={dateRange}
                                onChange={handleDateRangeChange}
                            >
                                <MenuItem value="week">Last week</MenuItem>
                                <MenuItem value="month">Last month</MenuItem>
                                <MenuItem value="three-month">Last 3 months</MenuItem>
                                <MenuItem value="max">Max</MenuItem>
                            </Select>
                        </Stack>

                        {/* Second Dropdown: Input Type */}
                        {showInputType && (
                            <Stack direction="row" alignItems="center">
                                <InputLabel sx={{ mr: 3 }}>Input Type:</InputLabel>
                                <Select
                                    value={inputType}
                                    onChange={handleInputTypeChange}
                                >
                                    <MenuItem value="static">Static</MenuItem>
                                    <MenuItem value="trace">Trace</MenuItem>
                                </Select>
                            </Stack>
                        )}
                    </Stack>

                    <Grid item xs={12}>
                        <AppMetricsDate
                            title={`Aggregated Metrics for ${metricName}`}
                            subheader={`Aggregated ${yaxis} Metrics (${dateRange})`}
                            metrics={periodMetrics}
                            dateArray={dateList}
                            yaxis={yaxis}
                        />
                    </Grid>
                    {cdf &&
                        <Stack direction="row" alignItems="center" sx={{ mb: 2, pt: 5, pl: 3 }}>
                            <InputLabel sx={{ mr: 3 }}>Date:</InputLabel>
                            <Select
                                value={selectedDate}
                                onChange={handleDateChange}
                                label="Select Date"
                                MenuProps={{
                                    PaperProps: {
                                        style: {
                                            maxHeight: 200, // Set max height of dropdown menu (in px)
                                            overflowY: "auto", // Enable vertical scrolling
                                        },
                                    },
                                }}
                            >
                                {dateList.map((date, index) => (
                                    <MenuItem key={index} value={date}>
                                        {`${date}`}
                                    </MenuItem>
                                ))}
                            </Select>
                        </Stack>
                    }

                    {cdf &&
                        <Grid item xs={12}>
                            <AppMetrics
                                title={metricName}
                                metricType={metricType}
                                subheader={`Latency vs CDF across all providers`}
                                metrics={metrics}
                                min={min}
                            />
                        </Grid>
                    }
                </Grid>
            </Container>
        </Page>
    );
};
AppMetricsPage.propTypes = {
    metricType: PropTypes.string.isRequired,
    streaming: PropTypes.bool,
    title: PropTypes.string,
    metricName: PropTypes.string.isRequired,
    min: PropTypes.number,
    cdf: PropTypes.bool
};
export default AppMetricsPage;
