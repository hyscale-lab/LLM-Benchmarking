import AppMetricsPage from "../sections/@dashboard/app/AppMetricsPage";

export default function TimeToFirstToken() {
  return (
    <div>
      <div style={{ paddingBottom: "30px" }}>
        <AppMetricsPage metricType="timetofirsttoken" title="Time To First Token Metrics" metricName="Time To First Token" min={500} cdf />
      </div>
      <div style={{ paddingBottom: "30px" }}>
        <AppMetricsPage metricType="timetofirsttoken_median" title="Time To First Token Metrics Median Metrics" metricName="Time To First Token Metrics Median" />
      </div>
      <div>
        <AppMetricsPage metricType="timetofirsttoken_p95" title="Time To First Token Metrics P95 Metrics" metricName="Time To First Token Metrics P95" />
      </div>
    </div>
  );
}
