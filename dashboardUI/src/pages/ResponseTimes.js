import AppMetricsPage from "../sections/@dashboard/app/AppMetricsPage";

export default function ResponseTimes() {
  const region = process.env.REACT_APP_AWS_REGION;
  return (
    <div>
      <div style={{ paddingBottom: "30px" }}>
        <AppMetricsPage metricType="response_times" title="Response Times Metrics" metricName="Response Times" min={500} cdf />
      </div>
      <div style={{ paddingBottom: "30px" }}>
        <AppMetricsPage metricType="response_times_median" title="Response Times Metrics Median Metrics" metricName="Response Times Metrics Median" />
      </div>
      <div>
        <AppMetricsPage metricType="response_times_p95" title="Response Times Metrics P95 Metrics" metricName="Response Times Metrics P95" />
      </div>
      <div>
        region: {region}
      </div>
    </div>
  );
}
