import { Grid, Card, Container, Stack, Box, Typography, CardContent, Link, TableRow, Paper, Table, TableCell, TableContainer, TableBody } from '@mui/material';
import ListItem from '@mui/material/ListItem';
// components
import Page from '../components/Page';

export default function LLMetrics() {
  return (
    <Page title="Dashboard: LLMetrics">
      <Container maxWidth="xl">
        <Stack direction="row" alignItems="center" justifyContent="space-between" mt={3}>
          {/* <Typography variant="h4" gutterBottom>
          LLMetrics
          </Typography> */}
        </Stack>

        <Grid item xs={12} mb={3}>
          <Card>
            <CardContent>
              <Typography variant="h4" marginBottom={3}>
                What is LLMetrics? (Large Language Model Metrics)
              </Typography>
              <Typography variant="p">
                LLMetrics is a comprehensive benchmarking tool designed to evaluate and compare the performance of Large Language Model (LLM) service providers. By standardizing the testing environment with fixed prompts, input tokens, and output tokens, LLMetrics ensures consistent and accurate performance assessments across providers. <br /><br />

                Our goal is to provide developers and organizations with actionable insights into the efficiency, responsiveness, and reliability of LLM providers, enabling them to make informed decisions for their applications.
              </Typography>

              <Typography variant="h5" marginTop={4} marginBottom={2}>
                Key Features
              </Typography>
              <Box sx={{ ml: 5, my: 2 }}>
                <ListItem sx={{ display: 'list-item' }}>
                  <b>Standardized Testing Environment:</b> Ensures consistency by using fixed prompts, input tokens, and output tokens across all providers.
                </ListItem>
                <ListItem sx={{ display: 'list-item' }}>
                  <b>Core Framework:</b> Built primarily in Python, LLMetrics includes dedicated classes for each provider with methods like <code>get_model_name</code>, <code>perform_inference_streaming</code>, <code>perform_inference</code>, and <code>log_metrics</code>.
                </ListItem>
                <ListItem sx={{ display: 'list-item' }}>
                  <b>Configuration Management:</b> The <code>main.py</code> script uses a JSON configuration to specify the provider, model, number of requests, input token count, max output size, and whether the request is streaming or non-streaming.
                </ListItem>
                <ListItem sx={{ display: 'list-item' }}>
                  <b>Data Persistence:</b> All benchmarking results are securely stored in a DynamoDB database for consistency, scalability, and easy retrieval.
                </ListItem>
                <ListItem sx={{ display: 'list-item' }}>
                  <b>Data Visualization:</b> Results are visualized through an interactive dashboard deployed on GitHub Pages, providing clear and actionable insights.
                </ListItem>
              </Box>

              <Typography variant="h5" marginTop={4} marginBottom={2}>
                Metrics Analyzed
              </Typography>
              <Box sx={{ ml: 5, my: 2 }}>
                <ListItem sx={{ display: 'list-item' }}>
                  <b>Time to First Token:</b> Measures how quickly the first token of a response is generated.
                </ListItem>
                <ListItem sx={{ display: 'list-item' }}>
                  <b>Time Between Tokens:</b> Tracks the interval between the generation of consecutive tokens. This metric is analyzed using:
                  <Box sx={{ ml: 3, mt: 1 }}>
                    <ListItem sx={{ display: 'list-item' }}>
                      <b>Median:</b> Represents the middle value of all recorded intervals, providing a measure of typical performance.
                    </ListItem>
                    <ListItem sx={{ display: 'list-item' }}>
                      <b>95th Percentile (p95):</b> Captures the upper bound of latency, ensuring that 95% of token intervals fall below this value. This metric is critical for identifying outliers and ensuring consistent performance.
                    </ListItem>
                  </Box>
                </ListItem>
                <ListItem sx={{ display: 'list-item' }}>
                  <b>Total Response Time:</b> Captures the overall time taken to generate the complete response.
                </ListItem>
              </Box>

              <Typography variant="h5" marginTop={4} marginBottom={2}>
                Benchmarked Providers
              </Typography>
              <Box sx={{ ml: 5, my: 2 }}>
                <ListItem sx={{ display: 'list-item' }}>Anthropic</ListItem>
                <ListItem sx={{ display: 'list-item' }}>AWS</ListItem>
                <ListItem sx={{ display: 'list-item' }}>Azure</ListItem>
                <ListItem sx={{ display: 'list-item' }}>Cloudflare</ListItem>
                <ListItem sx={{ display: 'list-item' }}>Google</ListItem>
                <ListItem sx={{ display: 'list-item' }}>Groq</ListItem>
                <ListItem sx={{ display: 'list-item' }}>Hyperbolic</ListItem>
                <ListItem sx={{ display: 'list-item' }}>OpenAI</ListItem>
                <ListItem sx={{ display: 'list-item' }}>Perplexity</ListItem>
                <ListItem sx={{ display: 'list-item' }}>Together AI</ListItem>
                <ListItem sx={{ display: 'list-item' }}>
                  <b>Local Server (VLLM):</b> A local server running the <Link target="_blank" href={'https://github.com/vllm-project/vllm'}>VLLM</Link> framework to serve LLM services, acting as a control for comparison with cloud providers.
                </ListItem>
              </Box>

              <Typography variant="h5" marginTop={4} marginBottom={2}>
                Models Used
              </Typography>
              <Typography variant="p">
                The following table summarizes the specific models being used for benchmarking by each provider.
              </Typography>
              <Box sx={{ my: 2 }}>
                <TableContainer component={Paper}>
                  <Table>
                    <TableBody>
                      <TableRow>
                        <TableCell><b>Provider</b></TableCell>
                        <TableCell><b>Model</b></TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>AWS Bedrock</TableCell>
                        <TableCell>meta.llama3-70b-instruct-v1:0</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Anthropic</TableCell>
                        <TableCell>claude-3-5-sonnet-20241022</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Azure</TableCell>
                        <TableCell>Meta-Llama-3.1-8B-Instruct</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Cloudflare</TableCell>
                        <TableCell>@cf/meta/llama-3.3-70b-instruct-fp8-fast</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Google Gemini</TableCell>
                        <TableCell>gemini-1.5-flash</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Groq</TableCell>
                        <TableCell>llama-3.3-70b-versatile</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Hyperbolic</TableCell>
                        <TableCell>meta-llama/Meta-Llama-3.1-70B-Instruct</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>OpenAI</TableCell>
                        <TableCell>gpt-4o</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Perplexity AI</TableCell>
                        <TableCell>sonar-pro</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Together AI</TableCell>
                        <TableCell>meta-llama/Llama-3.3-70B-Instruct-Turbo</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>vLLM (Local)</TableCell>
                        <TableCell>../scratch/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b</TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </Box>

              <Typography variant="h5" marginTop={4} marginBottom={2}>
                VLLM Implementation
              </Typography>
              <Typography variant="p">
                {/* vLLM Implementation Details */}
                In addition to benchmarking various cloud-based LLM service providers, LLMetrics incorporates a local deployment using vLLM to serve as a performance baseline. vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs, developed to optimize token generation and resource management.
                <br /><br />
                vLLM is integrated as a local provider to facilitate direct comparisons with cloud-based services. This integration involves the following steps:
                <Box sx={{ ml: 5, my: 2 }}>
                  <ol>
                    <li>
                      <strong>Setting Up vLLM Server:</strong> Deploy a vLLM server locally to handle inference requests, ensuring control over the environment and resources for consistent, reproducible benchmarking conditions.
                    </li>
                    <li>
                      <strong>Provider Class Implementation:</strong> Create a dedicated provider class within LLMetrics to interface with the vLLM server. This class implements essential methods such as <code>get_model_name</code>, <code>perform_inference_streaming</code>, <code>perform_inference</code>, and <code>log_metrics</code>, aligning with the structure used for other providers.
                    </li>
                    <li>
                      <strong>Configuration Management:</strong> Utilize the existing JSON configuration system in LLMetrics to specify parameters for the vLLM provider, including model selection, number of requests, input token count, maximum output size, and streaming preferences.
                    </li>
                  </ol>
                </Box>
              </Typography>
              <Typography variant="h5" marginTop={4} marginBottom={2}>
                Process Overview
              </Typography>
              <Typography variant="p">
                LLMetrics automates the benchmarking process through a robust infrastructure:
              </Typography>
              <Box sx={{ ml: 5, my: 2 }}>
                <ListItem sx={{ display: 'list-item' }}>
                  <b>Automation with GitHub Actions:</b> Weekly benchmarking runs are scheduled via GitHub Actions, ensuring regular and consistent evaluations.
                </ListItem>
                <ListItem sx={{ display: 'list-item' }}>
                  <b>CI Pipeline:</b> GitHub Actions also supports running CI tests and linters to maintain code quality and reliability.
                </ListItem>
                <ListItem sx={{ display: 'list-item' }}>
                  <b>Data Storage:</b> Experiment data is securely stored in a DynamoDB database for analysis and visualization.
                </ListItem>
                <ListItem sx={{ display: 'list-item' }}>
                  <b>Dashboard Deployment:</b> The interactive dashboard is deployed on GitHub Pages, pulling data from DynamoDB using Lambda functions.
                </ListItem>
              </Box>

              <Typography variant="h5" marginTop={4} marginBottom={2}>
                Infrastructure Overview
              </Typography>
              <Typography variant="p">
                LLMetrics leverages a robust infrastructure to ensure seamless benchmarking and visualization:
              </Typography>
              <Box sx={{ ml: 5, my: 2 }}>
                <ListItem sx={{ display: 'list-item' }}>
                  <b>GitHub Actions:</b> Automates weekly benchmarking runs and CI pipelines.
                </ListItem>
                <ListItem sx={{ display: 'list-item' }}>
                  <b>DynamoDB:</b> Securely stores experiment data for analysis.
                </ListItem>
                <ListItem sx={{ display: 'list-item' }}>
                  <b>Lambda Functions:</b> Facilitate data retrieval for the dashboard.
                </ListItem>
                <ListItem sx={{ display: 'list-item' }}>
                  <b>GitHub Pages:</b> Hosts the interactive dashboard for visualizing results.
                </ListItem>
              </Box>

              <Typography variant="p" marginTop={4}>
                LLMetrics is a part of the <Link target="_blank" href={'https://vhive-serverless.github.io/'}>vHive Ecosystem</Link>, a platform dedicated to advancing serverless computing and AI-driven solutions.
              </Typography>

              {/* Space for Image */}
              <Typography variant="h6" marginBottom={2} marginTop={4}>
                Infrastructure Diagram
              </Typography>
              <Box sx={{ my: 4, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>

                <img
                  src="/LLM-Benchmarking/static/icons/architecure.jpg"
                  alt="LLMetrics Infrastructure"
                  style={{ maxWidth: '100%', height: 'auto', borderRadius: '8px' }}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Container>
    </Page>
  );
}
