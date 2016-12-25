CARND_IP = '54.186.187.122'

namespace :carnd do
  task :ssh do
    sh "ssh carnd@#{CARND_IP}"
  end

  task :pull do
    sh "ssh -t carnd@#{CARND_IP} 'cd ~/carnd-behavioral-cloning && git pull'"
  end

  task :train, [:network] => :pull do |t, args|
    args.with_defaults(network: '~/carnd-keras-lab/networks/keras/german_traffic_sign_cnn.py')
    sh "ssh -t carnd@#{CARND_IP} 'python3 #{args[:network]}'"
  end

  task :drive do |t, args|
    sh 'python3 drive.py model.json'
  end

  task :scp, [:src, :dest] do |t, args|
    args.with_defaults(dest: '~')
    host = "carnd@#{CARND_IP}"
    puts "downloading #{args[:src]} to #{host}:#{args[:dest]}"
    sh "scp -rp #{args[:src]} #{host}:#{args[:dest]}"
  end

  task :sync, [:src, :dest] do |t, args|
    args.with_defaults(dest: '~')
    host = "carnd@#{CARND_IP}"
    puts "uploading #{args[:src]} to #{host}:#{args[:dest]}"

    sh "scp -rp driving_log.csv #{host}:~"
    sh "rsync -ravz --progress --ignore-existing IMG #{host}:~"

    unless args[:src].nil?
      sh "rsync -avvz --update --existing --ignore-existing #{args[:src]} #{host}:#{args[:dest]}"
    end
  end

  task :get_model, [] do
    host = "carnd@#{CARND_IP}"

    sh "rsync -avzh --progress #{host}:~/model.json ."
    sh "rsync -avzh --progress #{host}:~/model.h5 ."
  end

  task :down, [:src, :dest] do |t, args|
    args.with_defaults(dest: '~')
    host = "carnd@#{CARND_IP}"
    puts "downloading #{host}:#{args[:src]} to #{args[:dest]}"
    sh "scp -rp #{host}:#{args[:src]} #{args[:dest]}"
  end
end
